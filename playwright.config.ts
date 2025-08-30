import { defineConfig } from "@playwright/test";

export default defineConfig({
  timeout: 120000,
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
  use: {
    baseURL: "http://localhost:5000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    actionTimeout: 15000
  },
  webServer: {
command: 'DISABLE_STARTUP_GUARD=1 BACKGROUND_BASE_URL=http://localhost:5000 E2E_BASE_URL=http://localhost:5000 V4_MAX_RACES=50 V4_TREES=100 V4_MAX_DEPTH=12 V4_MIN_SAMPLES_LEAF=3 V4_CALIB_FOLDS=3 V4_SKIP_IMPORTANCE=1 PORT=5000 ./.venv/bin/python app.py --host 127.0.0.1 --port 5000',
    url: 'http://localhost:5000',
    reuseExistingServer: true,
    timeout: 240000
  },
  reporter: [["list"], ["html", { outputFolder: "tests/artifacts/playwright-report" }]]
});

