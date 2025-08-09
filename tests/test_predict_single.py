import requests
import json

# URL of the API endpoint
import os
url = f"http://localhost:{os.environ.get('DEFAULT_PORT', '5000')}/api/predict_single_race_enhanced"

# Headers
headers = {
    "Content-Type": "application/json"
}

# Sample data for POST request
payload = {
    "race_filename": "Race 6 - MEA - 26 July 2025.csv"  # Using actual file from unprocessed directory
}

# Send POST request
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Check if the request was successful
if response.ok:
    print("POST request successful. Status code:", response.status_code)
    print("Response:", response.json())
else:
    print("POST request failed. Status code:", response.status_code)
    print("Response:", response.text)

# Check logs for event completion
log_url = "http://localhost:5000/logs?query=prediction_completed"
log_response = requests.get(log_url)

# Check log response
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
