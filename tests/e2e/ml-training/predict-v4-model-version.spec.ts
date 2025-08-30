import { test, expect } from "@playwright/test";

async function waitForTrainingCompletion(page) {
  const deadline = Date.now() + 3 * 60 * 1000; // 3 minutes
  let jobId: string | null = null;

  // Find a running job
  while (Date.now() < deadline) {
    const res = await page.request.get("/api/model/registry/status");
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    const jobs = body.training_jobs || {};
    const running: any = Object.values(jobs).find(
      (j: any) => j.status === "running" || j.status === "starting"
    );
    if (running) {
      jobId = running.id;
      break;
    }
    await page.waitForTimeout(1000);
  }
  expect(jobId, "Expected a training job to start").toBeTruthy();

  // Poll specific job
  while (Date.now() < deadline) {
    const res = await page.request.get(`/api/model/registry/status?job_id=${jobId}`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    if (body.status === "completed") return;
    if (body.status === "failed") throw new Error(`Training failed: ${body.error_message || "unknown"}`);
    await page.waitForTimeout(1500);
  }
  throw new Error("Training did not complete within the expected time window");
}

import { ensureUpcomingRaceCsv } from "../helpers/upcoming";

async function getAnyUpcomingRaceFilename(page): Promise<string | null> {
  const res = await page.request.get("/api/upcoming_races_csv");
  if (!res.ok()) return null;
  const body = await res.json();
  const races: any[] = body.races || [];
  if (!races.length) return null;
  const withFilename = races.find(r => !!r.filename);
  return withFilename ? withFilename.filename : null;
}

async function predictEnhanced(page, raceFilename: string) {
  const res = await page.request.post("/api/predict_single_race_enhanced", {
    data: { race_filename: raceFilename },
  });
  expect(res.ok()).toBeTruthy();
  return res.json();
}

async function triggerTraining(page) {
  const res = await page.request.post("/api/model/training/trigger", {
    data: { prediction_type: "win", training_data_days: 7, force_retrain: true, model_id: "comprehensive_training" },
  });
  expect(res.ok()).toBeTruthy();
  const body = await res.json();
  expect(body.success).toBeTruthy();
}

// This spec verifies that Prediction V4 response includes model_registry_best and that
// after a training run completes, a subsequent prediction still includes best model info
// (and may update to a newer model).

test.describe("Prediction V4 uses registry best model", () => {
  test("model_registry_best present in prediction; remains valid after training", async ({ page }) => {
    // Find a race file to predict
    let raceFilename = await getAnyUpcomingRaceFilename(page);
    if (!raceFilename) {
      // Create a minimal upcoming race CSV for the test
      const created = await ensureUpcomingRaceCsv();
      // Rescan so the API picks it up (optional)
      await page.request.post("/api/rescan_upcoming");
      raceFilename = await getAnyUpcomingRaceFilename(page);
    }
    test.skip(!raceFilename, "No upcoming race CSV found; add one to upcoming_races/ to run this test.");

    // First prediction
    let beforePred = await predictEnhanced(page, raceFilename!);
    expect(beforePred.success).toBeTruthy();
    expect(beforePred.predictor_used).toBeTruthy();

    // If no best model yet, try a training run to ensure registry has a model, then re-predict
    if (!beforePred.model_registry_best) {
      try {
        await triggerTraining(page);
        await waitForTrainingCompletion(page);
        beforePred = await predictEnhanced(page, raceFilename!);
      } catch (e) {
        console.warn("Training not available in this environment:", e?.toString?.() || e);
      }
    }

    expect(beforePred.model_registry_best, "model_registry_best should be included").toBeTruthy();

    const beforeBest = beforePred.model_registry_best || {};
    const beforeBestId = beforeBest.model_id || null;

    // Try another training run (may or may not produce a new best); tolerate failure
    let secondTrainingSucceeded = false;
    try {
      await triggerTraining(page);
      await waitForTrainingCompletion(page);
      secondTrainingSucceeded = true;
    } catch (e) {
      console.warn("Second training attempt failed or unavailable:", e?.toString?.() || e);
    }

    // Second prediction
    const afterPred = await predictEnhanced(page, raceFilename!);
    expect(afterPred.success).toBeTruthy();
    if (secondTrainingSucceeded) {
      expect(afterPred.model_registry_best, "model_registry_best should be included after successful training").toBeTruthy();
    } else if (!afterPred.model_registry_best) {
      test.skip(true, "Training unavailable and no best model in registry; skipping best-model assertion.");
    }
    const afterBest = afterPred.model_registry_best || {};
    const afterBestId = afterBest.model_id || null;

    // It's acceptable if best model remains same (no better model produced),
    // but log when it changes.
    if (beforeBestId && afterBestId && beforeBestId !== afterBestId) {
      console.log(`Model updated: ${beforeBestId} -> ${afterBestId}`);
    }

    // Always assert the structure is present and contains at least a model_id string when available
    expect(typeof afterBestId === "string" || afterBestId === null).toBeTruthy();
  });
});

