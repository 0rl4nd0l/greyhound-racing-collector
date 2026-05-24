#!/usr/bin/env pwsh

# Test runner script for Greyhound Racing Dashboard (PowerShell version)
# This script starts the Flask app in testing mode and runs tests

param(
    [string]$TestType = "all",
    [int]$Port = 5002,
    [string]$Host = "localhost",
    [switch]$Headed,
    [switch]$Install,
    [switch]$Help
)

# Global variables
$FlaskProcess = $null

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

# Function to cleanup processes
function Stop-FlaskApp {
    if ($FlaskProcess -and !$FlaskProcess.HasExited) {
        Write-Status "Stopping Flask app (PID: $($FlaskProcess.Id))..."
        $FlaskProcess.Kill()
        $FlaskProcess.WaitForExit(5000)
    }
}

# Function to check if port is available
function Test-Port {
    param([int]$Port)
    
    $connection = Test-NetConnection -ComputerName $Host -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($connection) {
        Write-Error "Port $Port is already in use. Please stop the service or use a different port."
        exit 1
    }
}

# Function to start Flask app in testing mode
function Start-FlaskApp {
    Write-Status "Starting Flask app in testing mode on ${Host}:${Port}..."
    
    # Set testing environment variables
    $env:TESTING = "true"
    $env:FLASK_ENV = "testing"
    $env:MODULE_GUARD_STRICT = "0"
    $env:PREDICTION_IMPORT_MODE = "relaxed"
    
    # Check if virtual environment exists and activate it
    if (Test-Path ".venv") {
        Write-Status "Activating virtual environment (.venv)..."
        & ".venv\Scripts\Activate.ps1"
    } elseif (Test-Path "venv") {
        Write-Status "Activating virtual environment (venv)..."
        & "venv\Scripts\Activate.ps1"
    } else {
        Write-Warning "No virtual environment found. Using system Python."
    }
    
    # Start Flask app in background
    $FlaskProcess = Start-Process -FilePath "python" -ArgumentList "app.py", "--host", $Host, "--port", $Port -PassThru
    
    Write-Status "Flask app started with PID: $($FlaskProcess.Id)"
    Write-Status "Waiting for Flask app to be ready..."
    
    # Wait for Flask app to start (up to 30 seconds)
    $timeout = 30
    for ($i = 1; $i -le $timeout; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://${Host}:${Port}/ping" -UseBasicParsing -TimeoutSec 1
            if ($response.StatusCode -eq 200) {
                Write-Success "Flask app is ready!"
                return
            }
        } catch {
            # Continue waiting
        }
        
        if ($i -eq $timeout) {
            Write-Error "Flask app failed to start within $timeout seconds"
            exit 1
        }
        Start-Sleep -Seconds 1
    }
}

# Function to install dependencies
function Install-Dependencies {
    Write-Status "Installing Node.js dependencies..."
    npm install
    
    if (Get-Command playwright -ErrorAction SilentlyContinue) {
        Write-Status "Installing Playwright browsers..."
        npx playwright install
    }
}

# Function to run Cypress tests
function Start-CypressTests {
    Write-Status "Running Cypress tests..."
    if ($Headed) {
        npm run cypress:open
    } else {
        npm run cypress:run
    }
}

# Function to run Playwright tests
function Start-PlaywrightTests {
    Write-Status "Running Playwright tests..."
    if ($Headed) {
        npm run test:playwright:headed
    } else {
        npm run test:playwright
    }
}

# Function to run specific test file
function Start-SpecificTest {
    param(
        [string]$TestFile,
        [string]$Framework
    )
    
    Write-Status "Running specific test: $TestFile with $Framework"
    
    if ($Framework -eq "cypress") {
        npx cypress run --spec $TestFile
    } elseif ($Framework -eq "playwright") {
        npx playwright test $TestFile
    }
}

# Function to show help
function Show-Help {
    Write-Host "Usage: .\run-tests.ps1 [OPTIONS]"
    Write-Host "Options:"
    Write-Host "  -TestType TYPE      Test type (cypress, playwright, helper-routes, all)"
    Write-Host "  -Port PORT          Port for Flask app (default: 5002)"
    Write-Host "  -Host HOST          Host for Flask app (default: localhost)"
    Write-Host "  -Headed             Run tests in headed mode (with browser UI)"
    Write-Host "  -Install            Only install dependencies and exit"
    Write-Host "  -Help               Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run-tests.ps1                          # Run all tests"
    Write-Host "  .\run-tests.ps1 -TestType cypress        # Run only Cypress tests"
    Write-Host "  .\run-tests.ps1 -TestType helper-routes  # Run only helper routes tests"
    Write-Host "  .\run-tests.ps1 -Headed                  # Run tests with browser UI"
    Write-Host "  .\run-tests.ps1 -Port 5003               # Use port 5003"
}

# Main function
function Main {
    # Show help if requested
    if ($Help) {
        Show-Help
        exit 0
    }
    
    # Install dependencies if requested
    if ($Install) {
        Install-Dependencies
        exit 0
    }
    
    Write-Status "=== Greyhound Racing Dashboard Test Runner ==="
    Write-Status "Test Type: $TestType"
    Write-Status "Port: $Port"
    Write-Status "Host: $Host"
    Write-Status "Headed: $(!$Headed ? 'false' : 'true')"
    Write-Host ""
    
    # Register cleanup handler
    Register-EngineEvent PowerShell.Exiting -Action {
        Stop-FlaskApp
    }
    
    try {
        # Check if port is available
        Test-Port -Port $Port
        
        # Install dependencies if needed
        if (!(Test-Path "node_modules")) {
            Install-Dependencies
        }
        
        # Start Flask app
        Start-FlaskApp
        
        # Run tests based on type
        switch ($TestType.ToLower()) {
            "cypress" {
                Start-CypressTests
            }
            "playwright" {
                Start-PlaywrightTests
            }
            "helper-routes" {
                Write-Status "Running helper routes tests..."
                Start-SpecificTest -TestFile "cypress\e2e\test-helper-routes.cy.js" -Framework "cypress"
                Start-SpecificTest -TestFile "tests\playwright\test-helper-routes.spec.js" -Framework "playwright"
            }
            "all" {
                Write-Status "Running all tests..."
                Start-CypressTests
                Start-PlaywrightTests
            }
            default {
                Write-Error "Unknown test type: $TestType"
                Write-Error "Valid options: cypress, playwright, helper-routes, all"
                exit 1
            }
        }
        
        Write-Success "All tests completed successfully!"
    }
    finally {
        # Cleanup
        Stop-FlaskApp
    }
}

# Run main function
Main
