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

    // Emit a compact diagnostic snapshot when running headless
    // Helps diagnose CI differences without leaking secrets
    try {
      // eslint-disable-next-line no-console
      console.log('[API-only E2E] races summary:', races.map(r => ({
        venue: r.venue,
        race_date: r.race_date || r.date,
        filename: r.filename
      })).slice(0, 10));
    } catch {}

    // Be resilient to canonical naming and header overrides:
    // - accept either venue token "MEA"/"The Meadows" OR filename hint
    // - accept date from race_date or date field OR filename hint
    const found = races.find(r => {
      const venue = String(r.venue || '').toLowerCase();
      const date = String(r.race_date || r.date || '');
      const fname = String(r.filename || '').toLowerCase();
      const venueOk = venue.includes('mea') || venue.includes('meadows') || fname.includes('mea');
      const dateOk = date.includes('2025-08-22') || fname.includes('2025-08-22');
      return venueOk && dateOk;
    });
    expect(found).toBeTruthy();
  });
});

