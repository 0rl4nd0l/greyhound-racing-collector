// E2E API-only test: upcoming ingestion flow without UI navigation
import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import os from 'os';

const CSV_CONTENT = `Dog Name,BOX,WGT\nDog X,1,32.0\n`;

function getDownloadsDir(): string {
  // Prefer env, else default to ~/Downloads (matches server default in config.paths)
  const envDir = process.env.DOWNLOADS_WATCH_DIR;
  if (envDir && envDir.trim()) return path.resolve(envDir);
  return path.resolve(os.homedir(), 'Downloads');
}

test.describe('E2E API: Upcoming ingestion flow (no UI)', () => {
  test('ingest CSV and verify via /api/upcoming_races_csv', async ({ request }) => {
    // Warm up backend
    const h = await request.get('/api/health');
    expect(h.ok()).toBeTruthy();

    const downloadsDir = getDownloadsDir();
    if (!fs.existsSync(downloadsDir)) fs.mkdirSync(downloadsDir, { recursive: true });

    const fileName = 'Race 2 - MEA - 2025-08-22.csv';
    const csvPath = path.join(downloadsDir, fileName);

    // Simulate browser-like partial write then rename
    const partial = csvPath + '.part';
    fs.writeFileSync(partial, CSV_CONTENT, 'utf-8');
    fs.renameSync(partial, csvPath);

    // Ask backend to ingest downloads dir once
    const ing = await request.post('/api/dev/ingest_downloads_once');
    expect(ing.ok()).toBeTruthy();
    const ingJson = await ing.json();
    expect(ingJson.success).toBeTruthy();

    // Fetch upcoming races (CSV-based) and verify our test race is present by venue/date
    const resp = await request.get('/api/upcoming_races_csv?refresh=true&per_page=100');
    expect(resp.ok()).toBeTruthy();

    const data = await resp.json();
    expect(data.success).toBeTruthy();
    const races: any[] = Array.isArray(data.races) ? data.races : [];

    // Look for matching venue code or name and date
    const found = races.find(r => {
      const venue = String(r.venue || '').toLowerCase();
      const date = String(r.race_date || r.date || '');
      return (venue.includes('mea') || venue.includes('meadows')) && date.includes('2025-08-22');
    });
    expect(found).toBeTruthy();
  });
});

