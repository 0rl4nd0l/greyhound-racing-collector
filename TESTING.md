# Testing Guide for Greyhound Racing Dashboard

This document describes the testing setup for the Flask-based Greyhound Racing Dashboard, including Cypress and Playwright tests for the frontend components.

## Test Helper Routes

The application provides three special test helper routes that are only available when the Flask app is running in testing mode (when `TESTING=true` environment variable is set):

### 1. `/test-blank-page`
- **Purpose**: Provides a minimal HTML page with an empty container for injecting test scripts
- **Features**: Bootstrap CSS loaded, empty `#test-container` div ready for content injection
- **Use Case**: Testing custom JavaScript components in isolation

### 2. `/test-predictions`
- **Purpose**: Provides an HTML page with prediction results container and related JavaScript
- **Features**: Bootstrap, FontAwesome CSS, prediction-buttons.js script, hidden `#predictions-results-container`
- **Use Case**: Testing prediction display functionality and interaction logic

### 3. `/test-sidebar`
- **Purpose**: Provides a sidebar layout with Bootstrap grid system and sidebar.js functionality
- **Features**: Bootstrap grid layout, sidebar sections for logs/metrics/health, sidebar.js for real-time updates
- **Use Case**: Testing sidebar components, real-time updates, and responsive layout

## Testing Frameworks

### Cypress Tests
- **Location**: `cypress/e2e/test-helper-routes.cy.js`
- **Purpose**: End-to-end testing with real browser automation
- **Features**: 
  - DOM manipulation testing
  - Bootstrap CSS verification
  - Cross-browser compatibility
  - Navigation testing

### Playwright Tests  
- **Location**: `tests/playwright/test-helper-routes.spec.js`
- **Purpose**: Cross-browser testing with mobile/tablet/desktop viewports
- **Features**:
  - Multi-browser support (Chrome, Firefox, Safari)
  - Responsive design testing
  - Performance monitoring
  - Accessibility checks

## Prerequisites

1. **Node.js** (v14 or higher)
2. **Python** with Flask app dependencies
3. **Virtual Environment** (recommended: `.venv` or `venv`)

## Setup Instructions

### 1. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Playwright browsers (optional, but recommended)
npx playwright install
```

### 2. Using Test Runner Scripts

#### Unix/Linux/macOS (Bash):
```bash
# Make script executable
chmod +x run-tests.sh

# Run all tests
./run-tests.sh

# Run only Cypress tests
./run-tests.sh -t cypress

# Run only Playwright tests  
./run-tests.sh -t playwright

# Run only helper routes tests
./run-tests.sh -t helper-routes

# Run tests with browser UI (headed mode)
./run-tests.sh --headed

# Use custom port
./run-tests.sh -p 5003

# Show help
./run-tests.sh --help
```

#### Windows (PowerShell):
```powershell
# Run all tests
.\run-tests.ps1

# Run only Cypress tests
.\run-tests.ps1 -TestType cypress

# Run only Playwright tests
.\run-tests.ps1 -TestType playwright  

# Run only helper routes tests
.\run-tests.ps1 -TestType helper-routes

# Run tests with browser UI
.\run-tests.ps1 -Headed

# Use custom port
.\run-tests.ps1 -Port 5003

# Show help
.\run-tests.ps1 -Help
```

### 3. Manual Test Execution

If you prefer to run tests manually:

#### Start Flask App in Testing Mode:
```bash
# Set environment variables
export TESTING=true
export FLASK_ENV=testing

# Activate virtual environment
source .venv/bin/activate  # or source venv/bin/activate

# Start Flask app
python app.py --host localhost --port 5002
```

#### Run Cypress Tests:
```bash
# Headless mode
npm run cypress:run

# Interactive mode
npm run cypress:open

# Specific test file
npx cypress run --spec "cypress/e2e/test-helper-routes.cy.js"
```

#### Run Playwright Tests:
```bash
# Headless mode
npm run test:playwright

# With browser UI
npm run test:playwright:headed

# Interactive UI mode
npm run test:playwright:ui

# Specific test file
npx playwright test tests/playwright/test-helper-routes.spec.js
```

## Test Structure

### Test Categories

1. **Route Loading Tests**
   - Verify each helper route loads correctly
   - Check HTTP status codes
   - Validate basic HTML structure

2. **CSS Framework Tests**
   - Verify Bootstrap CSS is loaded and functional
   - Test responsive design across viewports
   - Check component styling

3. **JavaScript Functionality Tests**
   - Test script loading (sidebar.js, prediction-buttons.js)
   - Verify DOM manipulation capabilities
   - Test real-time update functionality

4. **Cross-Route Navigation Tests**
   - Test navigation between helper routes
   - Verify browser back/forward functionality
   - Check route state consistency

5. **Performance & Accessibility Tests**
   - Page load time monitoring
   - Basic accessibility compliance
   - Responsive design validation

### Test Data Scenarios

The tests simulate various scenarios:
- Empty/minimal content injection
- Real-time log updates with different severity levels
- Model metrics display with accuracy scores
- System health status updates
- Prediction results with confidence scores
- Error handling and graceful degradation

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find and kill process using port 5002
   lsof -ti:5002 | xargs kill -9
   ```

2. **Flask App Not Starting**
   - Check virtual environment activation
   - Verify Python dependencies installed
   - Check for conflicting environment variables

3. **Tests Timing Out**
   - Increase timeout values in test configs
   - Check network connectivity
   - Verify Flask app is responding to `/ping`

4. **Browser Issues (Playwright)**
   ```bash
   # Reinstall browsers
   npx playwright install --force
   ```

### Debug Mode

For debugging test failures:

```bash
# Cypress debug mode
npm run cypress:open

# Playwright debug mode  
npm run test:playwright:debug

# Run with headed browsers
./run-tests.sh --headed
```

## Configuration Files

- **`cypress.config.js`**: Cypress configuration
- **`playwright.config.js`**: Playwright configuration with multi-browser setup  
- **`package.json`**: Dependencies and test scripts
- **`run-tests.sh`**: Unix test runner script
- **`run-tests.ps1`**: PowerShell test runner script

## Contributing

When adding new test helper routes or modifying existing ones:

1. Update the Flask app with appropriate routes
2. Add corresponding test cases to both Cypress and Playwright
3. Update this documentation
4. Test on multiple browsers/viewports
5. Ensure tests pass in CI environment

## Continuous Integration

The test setup is designed to work in CI environments:

- Tests run headless by default
- Flask app starts automatically in testing mode
- Results output in standard formats (JSON, XML)
- Screenshots/videos captured on failure
- Multi-browser parallel execution supported

For CI integration, use:
```bash
# CI-optimized test execution
HEADLESS=true ./run-tests.sh -t all
```
