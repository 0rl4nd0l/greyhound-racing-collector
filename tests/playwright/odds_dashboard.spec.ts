import { test, expect } from '@playwright/test';

const base = process.env.BASE_URL || 'http://*********:5002';

// Focused E2E for Sportsbet Odds Dashboard only
// Verifies the page renders and backend endpoints respond successfully.
test.describe('Sportsbet Odds Dashboard', () => {
  test('renders dashboard and metrics, APIs respond OK', async ({ page, request }) => {
    // Visit the dashboard
    await page.goto(base + '/odds_dashboard', { waitUntil: 'networkidle' });

    // Basic page content
    await expect(page.locator('h1')).toContainText(/sportsbet odds dashboard/i);

    // Metrics and containers exist (support both legacy and new IDs)
    const racesMetricCount = await page.locator('#metric-races, #total-races').count();
    expect(racesMetricCount).toBeGreaterThan(0);

    const valuesMetricCount = await page.locator('#metric-values, #value-bets-count').count();
    expect(valuesMetricCount).toBeGreaterThan(0);

    const oddsContainerCount = await page.locator('#odds-summary, #live-odds-content').count();
    expect(oddsContainerCount).toBeGreaterThan(0);

    const valueBetsContainerCount = await page.locator('#value-bets, #value-bets-content').count();
    expect(valueBetsContainerCount).toBeGreaterThan(0);

    // API: live odds
    const liveRes = await request.get(base + '/api/sportsbet/live_odds');
    expect(liveRes.ok()).toBeTruthy();
    const live = await liveRes.json();
    expect(live).toHaveProperty('success', true);
    expect(live).toHaveProperty('odds_summary');

    // API: value bets
    const vbRes = await request.get(base + '/api/sportsbet/value_bets');
    expect(vbRes.ok()).toBeTruthy();
    const vb = await vbRes.json();
    expect(vb).toHaveProperty('success', true);
    expect(vb).toHaveProperty('value_bets');
  });
});

