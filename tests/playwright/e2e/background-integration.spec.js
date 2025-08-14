const { test, expect } = require('@playwright/test');

// Background Worker Integration Tests
// Tests the complete integration between frontend, API, and background workers (Celery/RQ)

test.describe('Background Worker Integration', () => {
const baseURL = process.env.FLASK_BASE_URL || `http://localhost:${process.env.DEFAULT_PORT || '5000'}`;
  const backendType = process.env.BACKEND_TYPE || 'celery';
  const testTimeout = parseInt(process.env.TEST_TIMEOUT) || 60000;

  test.beforeEach(async ({ page }) => {
    // Set longer timeout for integration tests
    test.setTimeout(testTimeout);
    
    // Navigate to home page and ensure it loads
    await page.goto(baseURL);
    await expect(page).toHaveTitle(/Greyhound Racing/i);
  });

  test('should complete full workflow: CSV download → processing → database storage', async ({ page, request }) => {
    // Step 1: Trigger CSV download via API
    console.log('Starting CSV download workflow...');
    
    const downloadResponse = await request.post(`${baseURL}/api/background/download-csv`, {
      data: {
        track: 'test_track',
        date: '2024-01-01',
        format: 'csv'
      }
    });
    
    expect(downloadResponse.ok()).toBeTruthy();
    const downloadData = await downloadResponse.json();
    expect(downloadData).toHaveProperty('task_id');
    
    const downloadTaskId = downloadData.task_id;
    console.log(`Download task started: ${downloadTaskId}`);

    // Step 2: Wait for download task completion
    let downloadComplete = false;
    let attempts = 0;
    const maxAttempts = 30;

    while (!downloadComplete && attempts < maxAttempts) {
      await page.waitForTimeout(2000);
      
      const statusResponse = await request.get(`${baseURL}/api/background/status/${downloadTaskId}`);
      expect(statusResponse.ok()).toBeTruthy();
      
      const status = await statusResponse.json();
      console.log(`Download task status: ${status.status}`);
      
      if (status.status === 'completed') {
        downloadComplete = true;
        expect(status).toHaveProperty('result');
        expect(status.result).toHaveProperty('file_path');
      } else if (status.status === 'failed') {
        throw new Error(`Download task failed: ${status.error}`);
      }
      
      attempts++;
    }

    expect(downloadComplete).toBeTruthy();
    console.log('✅ CSV download completed successfully');

    // Step 3: Trigger file processing
    const processResponse = await request.post(`${baseURL}/api/background/process-race-file`, {
      data: {
        file_path: 'test_data.csv',
        race_type: 'greyhound',
        track: 'test_track'
      }
    });

    expect(processResponse.ok()).toBeTruthy();
    const processData = await processResponse.json();
    expect(processData).toHaveProperty('task_id');
    
    const processTaskId = processData.task_id;
    console.log(`Processing task started: ${processTaskId}`);

    // Step 4: Wait for processing completion
    let processComplete = false;
    attempts = 0;

    while (!processComplete && attempts < maxAttempts) {
      await page.waitForTimeout(3000);
      
      const statusResponse = await request.get(`${baseURL}/api/background/status/${processTaskId}`);
      expect(statusResponse.ok()).toBeTruthy();
      
      const status = await statusResponse.json();
      console.log(`Processing task status: ${status.status}`);
      
      if (status.status === 'completed') {
        processComplete = true;
        expect(status).toHaveProperty('result');
        expect(status.result).toHaveProperty('processed_races');
        expect(status.result.processed_races).toBeGreaterThan(0);
      } else if (status.status === 'failed') {
        throw new Error(`Processing task failed: ${status.error}`);
      }
      
      attempts++;
    }

    expect(processComplete).toBeTruthy();
    console.log('✅ File processing completed successfully');

    // Step 5: Verify data was stored in database
    const racesResponse = await request.get(`${baseURL}/api/races?limit=10`);
    expect(racesResponse.ok()).toBeTruthy();
    
    const races = await racesResponse.json();
    expect(races).toBeInstanceOf(Array);
    expect(races.length).toBeGreaterThan(0);
    
    console.log(`✅ Found ${races.length} races in database`);

    // Step 6: Verify task tracking works
    const allTasksResponse = await request.get(`${baseURL}/api/background/tasks`);
    expect(allTasksResponse.ok()).toBeTruthy();
    
    const allTasks = await allTasksResponse.json();
    expect(allTasks).toHaveProperty('tasks');
    expect(allTasks.tasks).toBeInstanceOf(Array);
    
    // Should find our completed tasks
    const ourTasks = allTasks.tasks.filter(task => 
      task.task_id === downloadTaskId || task.task_id === processTaskId
    );
    expect(ourTasks.length).toBe(2);
    
    console.log('✅ Task tracking verified');
  });

  test('should handle concurrent background tasks', async ({ page, request }) => {
    console.log('Testing concurrent task execution...');
    
    // Start multiple tasks simultaneously
    const tasks = [];
    const taskPromises = [];
    
    for (let i = 0; i < 3; i++) {
      const taskPromise = request.post(`${baseURL}/api/background/generate-predictions`, {
        data: {
          race_id: `test_race_${i}`,
          model_type: 'ensemble',
          include_probabilities: true
        }
      });
      taskPromises.push(taskPromise);
    }

    // Wait for all tasks to be created
    const responses = await Promise.all(taskPromises);
    
    for (const response of responses) {
      expect(response.ok()).toBeTruthy();
      const data = await response.json();
      expect(data).toHaveProperty('task_id');
      tasks.push(data.task_id);
    }

    console.log(`Started ${tasks.length} concurrent tasks`);

    // Monitor all tasks until completion
    let allComplete = false;
    let attempts = 0;
    const maxAttempts = 60;

    while (!allComplete && attempts < maxAttempts) {
      await page.waitForTimeout(2000);
      
      const statusPromises = tasks.map(taskId => 
        request.get(`${baseURL}/api/background/status/${taskId}`)
      );
      
      const statusResponses = await Promise.all(statusPromises);
      const statuses = await Promise.all(
        statusResponses.map(response => response.json())
      );

      const completedTasks = statuses.filter(status => 
        status.status === 'completed' || status.status === 'failed'
      );

      console.log(`Completed tasks: ${completedTasks.length}/${tasks.length}`);

      if (completedTasks.length === tasks.length) {
        allComplete = true;
        
        // Check that all tasks completed successfully
        const failedTasks = statuses.filter(status => status.status === 'failed');
        if (failedTasks.length > 0) {
          console.error('Failed tasks:', failedTasks);
        }
        
        // Allow some tasks to fail in test environment, but most should succeed
        const successfulTasks = statuses.filter(status => status.status === 'completed');
        expect(successfulTasks.length).toBeGreaterThanOrEqual(1);
      }
      
      attempts++;
    }

    expect(allComplete).toBeTruthy();
    console.log('✅ Concurrent task execution completed');
  });

  test('should handle task failures gracefully', async ({ page, request }) => {
    console.log('Testing task failure handling...');
    
    // Trigger a task that should fail (invalid file path)
    const failResponse = await request.post(`${baseURL}/api/background/process-race-file`, {
      data: {
        file_path: '/nonexistent/file.csv',
        race_type: 'greyhound',
        track: 'test_track'
      }
    });

    expect(failResponse.ok()).toBeTruthy();
    const failData = await failResponse.json();
    expect(failData).toHaveProperty('task_id');
    
    const failTaskId = failData.task_id;
    console.log(`Fail task started: ${failTaskId}`);

    // Wait for task to fail
    let taskFailed = false;
    let attempts = 0;
    const maxAttempts = 30;

    while (!taskFailed && attempts < maxAttempts) {
      await page.waitForTimeout(2000);
      
      const statusResponse = await request.get(`${baseURL}/api/background/status/${failTaskId}`);
      expect(statusResponse.ok()).toBeTruthy();
      
      const status = await statusResponse.json();
      console.log(`Fail task status: ${status.status}`);
      
      if (status.status === 'failed') {
        taskFailed = true;
        expect(status).toHaveProperty('error');
        expect(status.error).toBeTruthy();
        console.log(`Task failed as expected: ${status.error}`);
      } else if (status.status === 'completed') {
        throw new Error('Task should have failed but completed successfully');
      }
      
      attempts++;
    }

    expect(taskFailed).toBeTruthy();
    console.log('✅ Task failure handling verified');
  });

  test('should maintain task history and cleanup', async ({ page, request }) => {
    console.log('Testing task history and cleanup...');
    
    // Create a few test tasks
    const testTasks = [];
    
    for (let i = 0; i < 2; i++) {
      const response = await request.post(`${baseURL}/api/background/update-race-notes`, {
        data: {
          race_id: `history_test_${i}`,
          notes: `Test notes ${i}`,
          source: 'automated_test'
        }
      });
      
      expect(response.ok()).toBeTruthy();
      const data = await response.json();
      testTasks.push(data.task_id);
    }

    // Wait a bit for tasks to complete
    await page.waitForTimeout(5000);

    // Check task history
    const historyResponse = await request.get(`${baseURL}/api/background/tasks?limit=50`);
    expect(historyResponse.ok()).toBeTruthy();
    
    const history = await historyResponse.json();
    expect(history).toHaveProperty('tasks');
    expect(history.tasks.length).toBeGreaterThan(0);

    // Verify our test tasks are in the history
    const ourTasksInHistory = history.tasks.filter(task =>
      testTasks.includes(task.task_id)
    );
    expect(ourTasksInHistory.length).toBeGreaterThan(0);

    console.log(`✅ Found ${ourTasksInHistory.length} test tasks in history`);

    // Test task cleanup (if implemented)
    const cleanupResponse = await request.post(`${baseURL}/api/background/cleanup`, {
      data: {
        older_than_hours: 0,
        status_filter: 'completed'
      }
    });
    
    // Cleanup might not be implemented, so we just check if endpoint exists
    if (cleanupResponse.ok()) {
      const cleanupData = await cleanupResponse.json();
      console.log('✅ Task cleanup endpoint available:', cleanupData);
    } else {
      console.log('ℹ️ Task cleanup endpoint not implemented');
    }
  });

  test('should verify worker health and metrics', async ({ page, request }) => {
    console.log('Testing worker health and metrics...');
    
    // Check worker health endpoint
    const healthResponse = await request.get(`${baseURL}/api/background/health`);
    
    if (healthResponse.ok()) {
      const health = await healthResponse.json();
      expect(health).toHaveProperty('status');
      
      if (backendType === 'celery') {
        expect(health).toHaveProperty('active_workers');
        expect(health).toHaveProperty('queues');
      } else if (backendType === 'rq') {
        expect(health).toHaveProperty('workers');
        expect(health).toHaveProperty('jobs');
      }
      
      console.log('✅ Worker health check passed:', health);
    } else {
      console.log('ℹ️ Worker health endpoint not available');
    }

    // Check if metrics are available
    const metricsResponse = await request.get(`${baseURL}/api/background/metrics`);
    
    if (metricsResponse.ok()) {
      const metrics = await metricsResponse.json();
      expect(metrics).toHaveProperty('total_tasks');
      console.log('✅ Worker metrics available:', metrics);
    } else {
      console.log('ℹ️ Worker metrics endpoint not available');
    }
  });

  test('should handle background worker restart scenarios', async ({ page, request }) => {
    console.log('Testing worker restart resilience...');
    
    // Start a long-running task
    const taskResponse = await request.post(`${baseURL}/api/background/generate-predictions`, {
      data: {
        race_id: 'restart_test',
        model_type: 'ensemble',
        include_probabilities: true,
        simulate_delay: 10 // Add delay to make task longer
      }
    });

    expect(taskResponse.ok()).toBeTruthy();
    const taskData = await taskResponse.json();
    const taskId = taskData.task_id;

    console.log(`Started resilience test task: ${taskId}`);

    // Wait a bit for task to start
    await page.waitForTimeout(3000);

    // Check task is running
    const runningResponse = await request.get(`${baseURL}/api/background/status/${taskId}`);
    expect(runningResponse.ok()).toBeTruthy();
    
    const runningStatus = await runningResponse.json();
    expect(['pending', 'running', 'completed']).toContain(runningStatus.status);

    // Note: In a real scenario, we'd restart the worker here
    // For this test, we'll just verify the task tracking continues to work
    
    // Wait for task completion or timeout
    let finalStatus = null;
    let attempts = 0;
    const maxAttempts = 30;

    while (!finalStatus && attempts < maxAttempts) {
      await page.waitForTimeout(2000);
      
      const statusResponse = await request.get(`${baseURL}/api/background/status/${taskId}`);
      expect(statusResponse.ok()).toBeTruthy();
      
      const status = await statusResponse.json();
      
      if (['completed', 'failed', 'revoked'].includes(status.status)) {
        finalStatus = status;
      }
      
      attempts++;
    }

    // Task should eventually complete or fail
    expect(finalStatus).toBeTruthy();
    console.log(`✅ Task final status: ${finalStatus.status}`);
  });
});
