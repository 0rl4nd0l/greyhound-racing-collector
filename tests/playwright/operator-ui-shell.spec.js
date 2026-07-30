const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

const route = '/operator-ui/prototype';
const areaIds = [
  'next-race',
  'collector-summary',
  'corpus-funnel',
  'model-identity',
  'recent-predictions',
  'system-health',
  'activity-feed',
];
const stateLabels = [
  'AVAILABLE/FRESH',
  'STALE',
  'UNAVAILABLE/DATA_MISSING',
  'WAITING',
  'RUNNING',
  'BLOCKED',
];

function computedTimesInMilliseconds(value) {
  return value.split(',').map((duration) => {
    const match = duration.trim().match(/^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(ms|s)$/i);
    expect(match, `valid computed time: ${duration}`).not.toBeNull();
    const milliseconds = Number(match[1]) * (match[2].toLowerCase() === 's' ? 1000 : 1);
    expect(Number.isFinite(milliseconds), `finite computed time: ${duration}`).toBe(true);
    return milliseconds;
  });
}

async function expectNoHorizontalOverflow(page) {
  const dimensions = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
}

test.describe('fixture-only operator console shell', () => {
  test('desktop exposes persistent warnings, landmarks, focus, and accessible components', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(route);

    await expect(page.getByText('PROTOTYPE DATA').first()).toBeVisible();
    await expect(page.getByText('RESEARCH ONLY — NOT FOR BETTING').first()).toBeVisible();
    await expect(page.getByRole('navigation', { name: 'Console sections' })).toBeVisible();
    await expect(page.getByRole('main')).toBeVisible();
    await expect(page.getByRole('contentinfo')).toBeVisible();
    await expect(page.locator('[data-dashboard-area]')).toHaveCount(7);
    for (const areaId of areaIds) {
      await expect(page.getByRole('navigation', { name: 'Console sections' }).locator(`a[href="#${areaId}"]`)).toHaveCount(1);
      const area = page.locator(`[data-dashboard-area="${areaId}"]`);
      await expect(area).toBeVisible();
      await expect(area.getByText('Updated at', { exact: true })).toHaveCount(1);
      await expect(area.getByText('Evidence source', { exact: true })).toHaveCount(1);
      expect(await area.getByText('PROTOTYPE DATA', { exact: true }).count()).toBeGreaterThanOrEqual(4);
    }
    for (const stateLabel of stateLabels) {
      await expect(page.getByText(stateLabel, { exact: true }).first()).toBeVisible();
    }
    const launch = page.getByRole('button', { name: 'Launch prediction' });
    await expect(launch).toBeDisabled();
    await expect(page.getByText('Disabled — prototype preview is not connected.')).toBeVisible();
    await expect(page.getByText('2099-04-01 · Sandown Park · Race 6 · Jump 09:30 UTC · Fixture race ID FIXTURE-RACE-20990401-SANDOWN-R06')).toBeVisible();
    await expectNoHorizontalOverflow(page);

    await page.keyboard.press('Tab');
    const skipLink = page.getByRole('link', { name: 'Skip to prototype content' });
    await expect(skipLink).toBeFocused();
    const focusOutline = await skipLink.evaluate((element) => getComputedStyle(element).outlineStyle);
    expect(focusOutline).not.toBe('none');

    const accessibility = await new AxeBuilder({ page }).analyze();
    expect(accessibility.violations).toEqual([]);
  });

  test('375px layout has no overflow and preserves warnings and navigation', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto(route);

    await expect(page.getByText('PROTOTYPE DATA').first()).toBeVisible();
    await expect(page.getByText('RESEARCH ONLY — NOT FOR BETTING').first()).toBeVisible();
    await expect(page.getByRole('navigation', { name: 'Console sections' })).toBeVisible();
    await expect(page.locator('[data-dashboard-area]')).toHaveCount(7);
    for (const areaId of areaIds) {
      await expect(page.getByRole('navigation', { name: 'Console sections' }).locator(`a[href="#${areaId}"]`)).toHaveCount(1);
      await expect(page.locator(`[data-dashboard-area="${areaId}"]`)).toBeVisible();
    }
    for (const stateLabel of stateLabels) {
      await expect(page.getByText(stateLabel, { exact: true }).first()).toBeVisible();
    }
    await expect(page.getByRole('button', { name: 'Launch prediction' })).toBeDisabled();
    await expectNoHorizontalOverflow(page);

    await page.locator('[data-dashboard-area="activity-feed"]').scrollIntoViewIfNeeded();
    await expect(page.getByText('RESEARCH ONLY — NOT FOR BETTING').first()).toBeInViewport();
    await expectNoHorizontalOverflow(page);
  });

  test('reduced motion preference disables smooth scrolling and animation', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto(route);
    expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true);
    const motion = await page.locator('html').evaluate((element) => {
      const panel = document.querySelector('.panel');
      return {
        scrollBehavior: getComputedStyle(element).scrollBehavior,
        animationDuration: getComputedStyle(panel).animationDuration,
        transitionDuration: getComputedStyle(panel).transitionDuration,
      };
    });

    expect(motion.scrollBehavior).toBe('auto');
    for (const duration of [
      ...computedTimesInMilliseconds(motion.animationDuration),
      ...computedTimesInMilliseconds(motion.transitionDuration),
    ]) {
      expect(duration).toBeLessThanOrEqual(0.01);
    }
  });
});
