import requests
import json
import os

# Manual test script for hitting a running server instance
# Note: This script is archived to avoid breaking automated pytest runs.
# To use it manually, run the Flask app and then execute this file.

url = f"http://localhost:{os.environ.get('DEFAULT_PORT', '5000')}/api/predict_single_race_enhanced"

headers = {"Content-Type": "application/json"}

payload = {
    "race_filename": "Race 6 - MEA - 26 July 2025.csv"
}

if __name__ == "__main__":
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
        if response.ok:
            print("POST request successful. Status code:", response.status_code)
            print("Response:", response.json())
        else:
            print("POST request failed. Status code:", response.status_code)
            print("Response:", response.text)

        log_url = f"http://localhost:{os.environ.get('DEFAULT_PORT', '5000')}/logs?query=prediction_completed"
        log_response = requests.get(log_url, timeout=10)
        if log_response.ok:
            logs = log_response.json()
            if logs:
                print("Successfully found prediction completion logs.")
                for log in logs:
                    print(log)
            else:
                print("No relevant logs found.")
        else:
            print("Failed to retrieve logs. Status code:", log_response.status_code)
    except Exception as e:
        print(f"Error contacting server: {e}")
