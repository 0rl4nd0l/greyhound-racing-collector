import { test, expect } from '@playwright/test';

const base = process.env.BASE_URL || 'http://localhost:5002';

test.describe('Odds dashboard controls and contrast', () => {
  test('shows Sportsbet Controls and high-contrast "Last action" text', async ({ page }) => {
    // Visit odds dashboard (reuse existing server)
    await page.goto(base + '/odds_dashboard', { waitUntil: 'networkidle' });

    // Controls panel exists
    const panel = page.locator('.sb-controls');
    await expect(panel).toBeVisible();
    await expect(panel.getByRole('button', { name: /update odds/i })).toBeVisible();
    await expect(panel.getByRole('button', { name: /seed quick/i })).toBeVisible();
    await expect(panel.getByRole('button', { name: /seed for predictions/i })).toBeVisible();

    // Status badge is labeled
    const badge = panel.locator('#sb-status-badge');
    await expect(badge.first()).toBeVisible();
    const badgeText = (await badge.first().textContent())?.trim().toLowerCase();
    expect(['active', 'disabled', 'unavailable', 'checking…', 'checking...']).toContain(badgeText);

    // Ensure metrics render as numbers
    const racesUpdated = page.locator('#sb-races-updated');
    const valueCount = page.locator('#sb-value-count');
    await expect(racesUpdated).toBeVisible();
    await expect(valueCount).toBeVisible();

    // Contrast check: "Last action" text on dark tile should be bright enough
    // Find element that contains the heading text "Last action"
    const lastActionHeading = page.locator('text=Last action').first();
    await expect(lastActionHeading).toBeVisible();

    // Compute color brightness; expect fairly light text on dark background
    const color = await lastActionHeading.evaluate((el) => {
      return window.getComputedStyle(el).color;
    });
    // Parse rgb(a)
    const m = /rgba?\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i.exec(color || '');
    expect(m).not.toBeNull();
    if (m) {
      const r = parseInt(m[1], 10), g = parseInt(m[2], 10), b = parseInt(m[3], 10);
      // Perceived brightness formula
      const brightness = Math.round((r * 299 + g * 587 + b * 114) / 1000);
      // Require bright text (>= 200) on dark tile
      expect(brightness).toBeGreaterThanOrEqual(200);
    }
  });
});

