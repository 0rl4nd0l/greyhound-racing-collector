import { defineConfig } from "@playwright/test";
import path from "path";

// Ensure tests default to port 5002 unless overridden
process.env.DEFAULT_PORT = process.env.DEFAULT_PORT || '5002';
process.env.FLASK_BASE_URL = process.env.FLASK_BASE_URL || 'http://localhost:5002';
process.env.BASE_URL = process.env.BASE_URL || 'http://localhost:5002';
// Ensure a stable downloads watch directory for E2E and enable watchers
const DL = process.env.DOWNLOADS_WATCH_DIR || path.resolve('tmp_e2e_downloads');
process.env.DOWNLOADS_WATCH_DIR = DL;
process.env.WATCH_DOWNLOADS = process.env.WATCH_DOWNLOADS || '1';
process.env.WATCH_UPCOMING = process.env.WATCH_UPCOMING || '1';

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
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: "http://localhost:5003",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    actionTimeout: 15000
  },
  webServer: {
command: 'PORT=5003 ./.venv/bin/python app.py --host 127.0.0.1 --port 5003',
    url: 'http://localhost:5003',
    reuseExistingServer: false,
    timeout: 240000,
    env: {
      TESTING: '1',
      TRAINING_MAX_SECS: '30',
      DISABLE_STARTUP_GUARD: '1',
      ENABLE_ENDPOINT_DROPDOWNS: '1',
      DISABLE_NAV_DROPDOWNS: '1',
      BACKGROUND_BASE_URL: 'http://localhost:5003',
      E2E_BASE_URL: 'http://localhost:5003',
      WATCH_DOWNLOADS: '1',
      WATCH_UPCOMING: '1',
      DOWNLOADS_WATCH_DIR: DL,
      UPCOMING_RACES_DIR: path.resolve('upcoming_races_temp'),
      V4_MAX_RACES: '50',
      V4_TREES: '100',
      V4_MAX_DEPTH: '12',
      V4_MIN_SAMPLES_LEAF: '3',
      V4_CALIB_FOLDS: '3',
      V4_SKIP_IMPORTANCE: '1'
    }
  },
  reporter: [["list"], ["html", { outputFolder: "tests/artifacts/playwright-report" }]]
});
