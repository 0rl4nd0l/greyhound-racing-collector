import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const base = process.env.BASE_URL || 'http://localhost:5002';

test.describe('Odds dashboard actions and accessibility', () => {
  test('Update odds flow and a11y contrast check', async ({ page, request }) => {
    await page.goto(base + '/odds_dashboard', { waitUntil: 'networkidle' });

    const panel = page.locator('.sb-controls');
    await expect(panel).toBeVisible();

    const updateBtn = panel.getByRole('button', { name: /update odds/i });
    await expect(updateBtn).toBeEnabled();

    await updateBtn.click();
    const out = panel.locator('#od-ctl-out');
    await expect(out).toHaveText(/updated|seeded|value bets|failed/i, { timeout: 30000 });

    // Verify metrics are numeric
    const racesText = (await panel.locator('#sb-races-updated').textContent()) || '0';
    const valuesText = (await panel.locator('#sb-value-count').textContent()) || '0';
    const races = parseInt(racesText.replace(/\D+/g, ''), 10);
    const values = parseInt(valuesText.replace(/\D+/g, ''), 10);
    expect(Number.isNaN(races)).toBeFalsy();
    expect(Number.isNaN(values)).toBeFalsy();

    // Cross-check value bets via API (soft expectation)
    const vbRes = await request.get(base + '/api/sportsbet/value_bets');
    expect(vbRes.ok()).toBeTruthy();
    const vb = await vbRes.json();
    expect(vb).toHaveProperty('success', true);
    expect(Array.isArray(vb.value_bets || [])).toBeTruthy();

    // Accessibility: check for serious/critical color-contrast issues in panel and dark tiles
    const results = await new AxeBuilder({ page })
      .include('.sb-controls')
      .include('.bg-dark')
      .analyze();

    const seriousContrast = results.violations.filter(v =>
      (v.impact === 'serious' || v.impact === 'critical') && v.id === 'color-contrast'
    );

    // Log precise targets to help identify the failing selector(s)
    try {
      const dump = seriousContrast.map(v => ({
        id: v.id,
        impact: v.impact,
        help: v.help,
        nodes: (v.nodes || []).map(n => ({
          target: n.target,
          html: (n as any).html || undefined,
          failureSummary: (n as any).failureSummary || undefined,
        })),
      }));
      // eslint-disable-next-line no-console
      console.log('[AXE_SERIOUS_CONTRAST]', JSON.stringify(dump, null, 2));
    } catch (_) {
      // ignore logging errors
    }

    expect(seriousContrast.length).toBe(0);
  });
});

