// E2E Playwright test: simulate clicking Upcoming Races and verify newly ingested race appears
// Assumes dev server is running and serving the frontend at baseURL configured in Playwright

import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

// Helper to write a CSV in a known watch Downloads dir and wait a bit
async function simulateDownload(csvPath: string, content: string) {
  // Simulate a browser writing a partial then renaming
  const partial = csvPath + '.part';
  fs.writeFileSync(partial, content, 'utf-8');
  fs.renameSync(partial, csvPath);
}

const CSV_CONTENT = `Race Date,Venue,Race Number,Dog Name,Box,Trainer
2025-08-22,MEA,2,Dog M,3,Trainer M
`;

// This test requires the backend to ingest from DOWNLOADS_WATCH_DIR into UPCOMING_RACES_DIR
// and the UI Upcoming page to surface the new race automatically or on refresh.

test.describe('E2E: Upcoming Races ingestion and UI flow', () => {
  test('user sees new race in predictions after clicking Upcoming Races', async ({ page }) => {
    // Warm up backend to ensure server is ready
    try {
      await page.request.get('/api/health');
    } catch {}
    // Navigate to Upcoming page first
    await page.goto('/upcoming', { waitUntil: 'domcontentloaded' });

    // Prepare a temp downloads dir in the test environment if provided via env
    const downloadsDir = process.env.DOWNLOADS_WATCH_DIR || path.resolve('tmp_e2e_downloads');
    if (!fs.existsSync(downloadsDir)) fs.mkdirSync(downloadsDir, { recursive: true });

// Simulate a new CSV showing up in Downloads
const fileName = 'Race 2 - MEA - 2025-08-22.csv';
const csvPath = path.join(downloadsDir, fileName);
await simulateDownload(csvPath, CSV_CONTENT);

// Ask backend to ingest downloads once (test-only helper)
const resp = await page.request.post('/api/dev/ingest_downloads_once');
const j = await resp.json();
if (!j.success) {
  console.warn('ingest_downloads_once failed', j);
}

// Ask UI to refresh upcoming list without full navigation
await page.evaluate(() => (window as any).reloadUpcomingRaces && (window as any).reloadUpcomingRaces());

    // Now assert the new race appears (venue code or name and date on page)
    const foundVenue = await page.locator('text=/\\b(MEA|The Meadows)\\b/').first();
    await expect(foundVenue).toBeVisible();

    // Verify the date appears on the page
    await expect(page.locator('text=2025-08-22')).toBeVisible();
  });
});

