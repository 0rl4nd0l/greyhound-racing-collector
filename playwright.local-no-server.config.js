// @ts-check
const { defineConfig, devices } = require('@playwright/test');

// Local Playwright config that assumes the server is already running.
// It does not start a webServer and points to the local app on port 5002 by default.
module.exports = defineConfig({
  testDir: './tests',
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
  reporter: [['list'], ['html', { outputFolder: 'playwright-report' }]],
  use: {
    baseURL: process.env.BASE_URL || 'http://127.0.0.1:5002',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium-mobile',
      use: { ...devices['iPhone 12'], viewport: { width: 375, height: 812 } },
    },
    {
      name: 'chromium-tablet',
      use: { ...devices['Desktop Chrome'], viewport: { width: 768, height: 1024 } },
    },
    {
      name: 'chromium-desktop',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 800 } },
    },
  ],
});
