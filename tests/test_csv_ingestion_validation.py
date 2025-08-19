import requests
import logging

BASE_URL = "http://localhost:5002"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_logs")



def log_error(error_message):
    """Log errors to a specific error log"""
    logger.error(error_message)



def test_missing_file():
    """Test API response for missing file scenario."""
    response = requests.post(
        f"{BASE_URL}/api/ingest_csv",
        files={"file": ("does_not_exist.csv", None)}
    )
    
    assert response.status_code == 400
    assert response.json() == {"success": False, "error": "No file part"}


def test_malformed_csv():
    """Test API response for malformed CSV content."""

    corrupted_content = b"corrupted data"
    corrupted_file_path = "./corrupted_files/corrupted.csv"

    with open(corrupted_file_path, "wb") as f:
        f.write(corrupted_content)

    with open(corrupted_file_path, "rb") as file:
        response = requests.post(
            f"{BASE_URL}/api/ingest_csv",
            files={"file": ("corrupted.csv", file)}
        )

    assert response.status_code in [400, 422]
    assert "Could not parse file" in response.text

    # Confirm proper logging
    with open("logs/errors.log") as log_file:
        logs = log_file.read()
        assert "schema_mismatch" in logs
        assert "Traceback" not in logs


def main():
    """Main function to execute tests."""
    test_missing_file()
    test_malformed_csv()


if __name__ == "__main__":
    main()
