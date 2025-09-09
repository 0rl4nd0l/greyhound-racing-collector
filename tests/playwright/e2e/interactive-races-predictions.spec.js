// Playwright E2E: Verify prediction results container is auto-created and renders
// This test programmatically triggers displayPredictionResult using the
// PredictionButtonManager so it does not depend on data presence on the page.

const { test, expect } = require('@playwright/test');

// Helper: robustly navigate to the interactive races page
async function gotoInteractiveRaces(page) {
  await page.goto('/interactive-races', { waitUntil: 'load' });
  // Soft check: if server returned an error page, still proceed to attempt programmatic render
}

// Programmatic render path (does not rely on presence of Predict buttons)
// Validates our fix that creates the container if missing and renders content.
 test('renders programmatically and auto-creates prediction results container', async ({ page }) => {
  await gotoInteractiveRaces(page);

  // Proceed regardless of initial container state; we will ensure it becomes visible with content
  // Programmatically render a synthetic successful result using the module
  await page.waitForLoadState('domcontentloaded');

  // Some pages instantiate PredictionButtonManager on DOMContentLoaded; if not, create it.
  await page.evaluate(() => {
    try {
      if (typeof window.PredictionButtonManager !== 'undefined') {
        if (!window.__PB) {
          window.__PB = new window.PredictionButtonManager();
        }
      }
    } catch (e) { /* ignore */ }
  });

  // Drive the display using the public rendering method with a minimal payload
  await page.evaluate(() => {
    const result = {
      success: true,
      predictions: [
        { dog_name: 'Dog A', final_score: 0.42, box_number: 1 },
        { dog_name: 'Dog B', final_score: 0.21, box_number: 2 }
      ]
    };
    if (window.__PB && typeof window.__PB.displayPredictionResult === 'function') {
      window.__PB.displayPredictionResult(result);
    } else if (typeof window.PredictionButtonManager !== 'undefined') {
      const mgr = new window.PredictionButtonManager();
      mgr.displayPredictionResult(result);
      window.__PB = mgr;
    } else {
      // As a last resort, try calling the function that the module adds in global scope (if any)
      if (typeof window.displayPredictionResults === 'function') {
        window.displayPredictionResults([result]);
      }
    }
  });

  // Now the container should exist and include our top pick
  const container = page.locator('#prediction-results-container');
  await expect(container).toHaveCount(1);
  await expect(container).toContainText('Dog A');
});
