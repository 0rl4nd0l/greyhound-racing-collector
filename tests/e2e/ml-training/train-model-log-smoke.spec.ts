import { test, expect } from "@playwright/test";

// Quick smoke to verify the Training Progress & Logs panel updates soon after starting
// a registry-backed training job. We are not waiting for full job completion here.

test.describe("ML Training Page - log smoke", () => {
  test("log panel receives updates after starting registry training", async ({ page }) => {
    // Allow a bit more than default for initial server warm-up + first polls
    test.setTimeout(45000);

    await page.goto("/ml-training");

    // Click the registry-backed trigger
    const trainBtn = page.locator("button:has-text('Train New Model')");
    await expect(trainBtn).toHaveCount(1);
    await trainBtn.click();

    // Progress bar should appear
    const progressBar = page.locator("#training-progress-bar");
    await expect(progressBar).toBeVisible({ timeout: 10000 });

    // Logs container should start receiving lines shortly after
    const logs = page.locator("#training-logs");

    // Wait for logs to be cleared and then appended (innerHTML becomes something other than the placeholder)
    await page.waitForFunction(() => {
      const el = document.getElementById("training-logs");
      if (!el) return false;
      const txt = (el.textContent || "").trim();
      // Avoid initial placeholder text
      return !!txt && !/Training logs will appear here/i.test(txt);
    }, { timeout: 20000 });

    // At least one log line should be present (we append div per line)
    const anyLine = logs.locator("div").first();
    await expect(anyLine).toBeVisible({ timeout: 5000 });

    // Optionally assert status text has been updated
    const statusEl = page.locator("#training-status");
    await expect(statusEl).toContainText(/Status:|Training|Running|Progress|spinner/i, { timeout: 5000 });
  });
});

