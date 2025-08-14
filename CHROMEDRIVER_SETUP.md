# ChromeDriver + Selenium Stack Setup

## Overview

This document describes the ChromeDriver and Selenium stack setup implemented for the Greyhound Racing Predictor application.

## Components

### 1. Requirements (`requirements.txt`)
- **selenium==4.34.2**: Pinned to stable 4.x version
- **webdriver-manager==4.0.2**: Auto-downloads and manages ChromeDriver

### 2. Driver Helper (`drivers.py`)
- `get_chrome_driver(headless=True)`: Sets up Chrome WebDriver with optimized options
- `setup_selenium_driver_path()`: Sets SELENIUM_DRIVER_PATH environment variable
- Uses webdriver-manager for automatic ChromeDriver download and versioning

### 3. Updated Scrapers
- `hybrid_odds_scraper.py`: Updated to use the new driver helper
- Automatic ChromeDriver setup with fallback handling
- Enhanced Chrome options for better compatibility

### 4. Smoke Test (`tests/test_chromedriver_smoke.py`)
- Comprehensive test suite to verify ChromeDriver functionality
- Tests opening `about:blank` in headless Chrome
- Validates driver initialization, page loading, and cleanup
- Checks environment variable setup

### 5. CI/CD Integration (`.github/workflows/ci.yml`)
- Added ChromeDriver installation step: `python -m webdriver_manager.chrome`
- Integrated smoke test into CI pipeline
- Caches ChromeDriver for faster subsequent runs

### 6. Docker Support (`Dockerfile`)
- Multi-stage build with Chrome browser installation
- Pre-caches ChromeDriver using webdriver-manager
- Health check using driver setup verification
- Optimized for both development and production environments

## Usage

### Basic Usage
```python
from drivers import get_chrome_driver

# Get a headless Chrome driver
driver = get_chrome_driver(headless=True)
driver.get("https://example.com")
# ... use driver
driver.quit()
```

### Environment Setup
```python
from drivers import setup_selenium_driver_path

# Set up environment variable
driver_path = setup_selenium_driver_path()
print(f"ChromeDriver path: {driver_path}")
```

### Running Tests
```bash
# Run smoke test
python tests/test_chromedriver_smoke.py

# Run in CI environment
python -m webdriver_manager.chrome  # Pre-cache driver
python tests/test_chromedriver_smoke.py
```

## Features

### Chrome Options
- **Headless mode**: Runs without GUI for CI/production
- **No sandbox**: Required for containerized environments
- **Disable dev-shm-usage**: Prevents memory issues in containers
- **Disable GPU**: Optimizes for headless environments
- **Custom user agent**: Better compatibility with websites
- **Window size**: Set to 1920x1080 for consistent rendering

### Error Handling
- Automatic fallback mechanisms in hybrid scraper
- Proper driver cleanup on exceptions
- Comprehensive logging for debugging

### Performance Optimizations
- **Disable images**: Faster page loading
- **Disable JavaScript**: For simple content extraction
- **Disable extensions/plugins**: Reduced overhead

## Verification

The setup can be verified by running:
```bash
python tests/test_chromedriver_smoke.py
```

This will:
1. Download and cache the appropriate ChromeDriver
2. Test headless and non-headless modes
3. Verify environment variable setup
4. Confirm basic WebDriver functionality

## Integration Points

### Existing Scrapers
All existing Selenium-based scrapers should be updated to use:
```python
from drivers import get_chrome_driver, setup_selenium_driver_path
```

### CI/CD Pipeline
The CI now includes:
- Automatic ChromeDriver installation
- Smoke test execution
- Caching for performance

### Docker Deployment
The Dockerfile includes:
- System dependencies for Chrome
- Pre-cached ChromeDriver
- Health checks for driver availability

## Benefits

1. **Automatic Management**: No manual ChromeDriver downloads or version management
2. **Cross-Platform**: Works on macOS, Linux, Windows
3. **CI/CD Ready**: Integrated into build pipeline
4. **Docker Compatible**: Optimized for containerized deployments
5. **Fallback Support**: Graceful degradation when drivers fail
6. **Performance Optimized**: Minimal resource usage for headless operation

## Troubleshooting

### Common Issues
1. **Chrome not found**: Ensure Chrome browser is installed
2. **Permission errors**: Check file permissions on driver cache
3. **Network issues**: webdriver-manager requires internet access for initial download
4. **Container issues**: Ensure all Chrome dependencies are installed in Dockerfile

### Debug Mode
Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This provides detailed information about ChromeDriver download and setup processes.
