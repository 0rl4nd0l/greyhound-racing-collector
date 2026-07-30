const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

const route = '/operator-ui';

async function expectNoOverflow(page) {
  const width = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    client: document.documentElement.clientWidth,
  }));
  expect(width.scroll).toBeLessThanOrEqual(width.client);
}

test.describe('atomic fixture-only operator workflow', () => {
  test('golden flow selects, confirms, reconnects, and exposes evidence', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto(route);
    await expect(page.getByText('RESEARCH ONLY — NOT FOR BETTING').first()).toBeVisible();
    await page.getByRole('link', { name: 'Choose an exact race' }).click();
    await expect(page.getByRole('heading', { name: 'Date → meeting → race' })).toBeVisible();
    await expect(page.getByText('1 Apr 2099, 8:30 pm AEST')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Review research-only confirmation' })).toBeEnabled();
    await page.getByRole('button', { name: 'Review research-only confirmation' }).click();
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText('RESEARCH ONLY — NOT FOR BETTING')).toBeVisible();
    await page.keyboard.press('Tab');
    await page.getByRole('button', { name: 'Confirm fixture lifecycle' }).click();
    await expect(page.getByRole('heading', { name: 'One request, one attempt, no retry' })).toBeFocused();
    await expect(page.getByText('FIXTURE-JOB-20990401-0006')).toBeVisible();
    await page.reload();
    await expect(page.getByText('FIXTURE-JOB-20990401-0006')).toBeVisible();
    await page.getByRole('link', { name: 'View ranked probabilities' }).click();
    await expect(page.getByRole('heading', { name: 'Ranked win probabilities' })).toBeVisible();
    await expect(page.getByText('bundle-fixture-20990401-r06')).toBeVisible();
    await expectNoOverflow(page);
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  });

  test('all invalid selection fixtures fail closed with exact distinctions', async ({ page }) => {
    await page.goto(route);
    const selector = page.locator('#fixture-state');
    const review = page.getByRole('button', { name: 'Review research-only confirmation' });
    const fixtures = {
      ambiguous: 'ambiguous race identity',
      'post-jump': 'scheduled jump has passed',
      'missing-runner': 'active runner identity is missing',
      stale: '300-second pre-jump freshness policy',
      'missing-jump': 'scheduled jump identity is missing',
      unsupported: 'window or model configuration is unsupported',
      conflicting: 'source and selected-race evidence conflict',
      unavailable: 'source evidence is unavailable',
    };
    for (const [value, reason] of Object.entries(fixtures)) {
      await selector.selectOption(value);
      await expect(review).toBeDisabled();
      await expect(page.getByRole('status')).toContainText(reason);
    }
    await selector.selectOption('valid');
    await expect(review).toBeEnabled();
  });

  test('mobile contains long identities and preserves navigation and actions', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto(route);
    await expect(page.getByRole('navigation', { name: 'Console sections' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Review research-only confirmation' })).toBeVisible();
    await page.getByText('90766b65ba7f184d53b57c520fd9af1962797c9370984769d93eecc631716cea').scrollIntoViewIfNeeded();
    await expectNoOverflow(page);
  });

  test('keyboard dialog focus and print evidence view are supported', async ({ page }) => {
    await page.goto(route);
    await page.keyboard.press('Tab');
    await expect(page.getByRole('link', { name: 'Skip to prototype content' })).toBeFocused();
    await page.getByRole('button', { name: 'Review research-only confirmation' }).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('dialog')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).toBeHidden();
    await page.emulateMedia({ media: 'print' });
    await expect(page.locator('.sidebar')).toHaveCSS('display', 'none');
    await expect(page.locator('.evidence-view').first()).toBeVisible();
  });
});
