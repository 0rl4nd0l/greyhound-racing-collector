// @ts-check
const { defineConfig, devices } = require('@playwright/test');

/**
 * @see https://playwright.dev/docs/test-configuration
 */
module.exports = defineConfig({
  testDir: './tests',
  // Only run Playwright E2E specs; ignore unit/integration Jest tests
  testMatch: [
    'e2e/**/*.spec.@(ts|js)',
    'playwright/**/*.spec.@(ts|js)'
  ],
  testIgnore: [
    'unit/**',
    'integration/**',
    '**/*.test.js',
    '**/*.test.ts'
  ],
  
  /* Run tests in files in parallel */
  fullyParallel: false,
  
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  
  /* Opt out of parallel tests on CI. */
  workers: 1,
  
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'playwright-results.json' }]
  ],
  
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: `http://127.0.0.1:${process.env.PORT || '5002'}`,
    
    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'on-first-retry',
    
    /* Screenshot settings */
    screenshot: 'only-on-failure',
    
    /* Video settings */
    video: 'retain-on-failure',
  },

  /* Configure projects for major browsers with different viewport sizes */
  projects: [
    {
      name: 'chromium-mobile',
      use: { 
        ...devices['iPhone 12'],
        viewport: { width: 375, height: 812 }
      },
    },
    {
      name: 'chromium-tablet',
      use: { 
        ...devices['Desktop Chrome'],
        viewport: { width: 768, height: 1024 }
      },
    },
    {
      name: 'chromium-desktop',
      use: { 
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 }
      },
    },
    {
      name: 'firefox-mobile',
      use: { 
        ...devices['Desktop Firefox'],
        viewport: { width: 375, height: 812 }
      },
    },
    {
      name: 'firefox-tablet',
      use: { 
        ...devices['Desktop Firefox'],
        viewport: { width: 768, height: 1024 }
      },
    },
    {
      name: 'firefox-desktop',
      use: { 
        ...devices['Desktop Firefox'],
        viewport: { width: 1280, height: 720 }
      },
    },
    {
      name: 'webkit-mobile',
      use: { 
        ...devices['Desktop Safari'],
        viewport: { width: 375, height: 812 }
      },
    },
    {
      name: 'webkit-tablet',
      use: { 
        ...devices['Desktop Safari'],
        viewport: { width: 768, height: 1024 }
      },
    },
    {
      name: 'webkit-desktop',
      use: { 
        ...devices['Desktop Safari'],
        viewport: { width: 1280, height: 720 }
      },
    },
  ],

  /* Run your local dev server before starting the tests */
  webServer: {
    command: 'PORT=5002 ./.venv/bin/python app.py --host ********* --port 5002',
    url: 'http://*********:5002',
    reuseExistingServer: !process.env.CI,
    timeout: 180000,
    env: {
      FLASK_ENV: 'testing',
      MODULE_GUARD_STRICT: '0',
      PREDICTION_IMPORT_MODE: 'relaxed',
      ENABLE_ENDPOINT_DROPDOWNS: '1',
      TESTING: '1',
'TRAINING_MAX_SECS': '30',
      'DISABLE_NAV_DROPDOWNS': '1'
    }
  },
});
