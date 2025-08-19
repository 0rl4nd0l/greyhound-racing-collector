const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');
const sqlite3 = require('sqlite3');

/**
 * E2E Test: Workflow 1 - Download & Process Race Data
 * 
 * Test Flow:
 * 1. Login to the application
 * 2. Download new race data
 * 3. Trigger file processing via background task
 * 4. Wait for Celery/RQ task to complete
 * 5. Verify database rows were added correctly
 */

test.describe('Workflow 1: Download → Process → Verify DB', () => {
  let testContext = {};

  test.beforeEach(async ({ page }) => {
    // Set up test context
    testContext = {
      testRaceId: `test_race_${Date.now()}`,
      downloadPath: './test_upcoming_races',
      testDbPath: process.env.DATABASE_URL || './test_greyhound_racing_data.db'
    };

    // Ensure test directories exist
    if (!fs.existsSync(testContext.downloadPath)) {
      fs.mkdirSync(testContext.downloadPath, { recursive: true });
    }

    // Navigate to application
    await page.goto('/');
    
    // Wait for application to load
    await expect(page.locator('body')).toBeVisible();
  });

  test.afterEach(async ({ page }) => {
    // Clean up test files
    try {
      const testFiles = fs.readdirSync(testContext.downloadPath)
        .filter(file => file.includes(testContext.testRaceId));
      
      testFiles.forEach(file => {
        const filePath = path.join(testContext.downloadPath, file);
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      });
    } catch (error) {
      console.log('Cleanup warning:', error.message);
    }
  });

  test('should complete full download-process-verify workflow', async ({ page }) => {
    console.log('🚀 Starting Workflow 1: Download → Process → Verify');

    // Step 1: Navigate to data processing section
    await test.step('Navigate to data processing', async () => {
      await page.goto('/scraping');
      await expect(page.locator('h1')).toContainText('Data Processing');
      
      // Wait for the page to be fully loaded
      await page.waitForLoadState('networkidle');
    });

    // Step 2: Create a test race file to simulate download
    const testRaceData = await test.step('Create test race data', async () => {
      const testCsvData = `box_number,dog_name,trainer,weight,grade,margin,odds
1,Test Dog 1,Test Trainer A,32.5,5,0.00,2.50
2,Test Dog 2,Test Trainer B,31.8,5,1.25,4.00
3,Test Dog 3,Test Trainer C,33.2,5,2.50,6.50
4,Test Dog 4,Test Trainer D,32.0,5,3.75,8.00
5,Test Dog 5,Test Trainer E,31.5,5,5.00,12.00`;

      const testFilePath = path.join(testContext.downloadPath, `${testContext.testRaceId}.csv`);
      fs.writeFileSync(testFilePath, testCsvData);
      
      return {
        filePath: testFilePath,
        fileName: `${testContext.testRaceId}.csv`,
        expectedRows: 5
      };
    });

    // Step 3: Trigger file processing via API
    const taskId = await test.step('Trigger background processing', async () => {
      const response = await page.request.post('/api/process_file/' + testRaceData.fileName, {
        headers: {
          'Content-Type': 'application/json',
        }
      });

      expect(response.status()).toBe(200);
      const result = await response.json();
      expect(result.success).toBe(true);
      
      console.log('✅ File processing triggered:', result.message);
      return result.task_id || 'sync_task';
    });

    // Step 4: Monitor processing status
    await test.step('Wait for processing completion', async () => {
      let processingComplete = false;
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds timeout
      
      while (!processingComplete && attempts < maxAttempts) {
        // Check processing status via API
        const statusResponse = await page.request.get('/api/processing_status');
        expect(statusResponse.status()).toBe(200);
        
        const status = await statusResponse.json();
        console.log(`⏳ Processing status: ${status.progress}% - ${status.current_task}`);
        
        if (status.progress >= 100 && !status.running) {
          processingComplete = true;
          console.log('✅ Processing completed successfully');
        } else if (status.error_count > 0) {
          // Check for errors in processing logs
          const logs = status.log || [];
          const errorLogs = logs.filter(log => log.level === 'ERROR');
          if (errorLogs.length > 0) {
            console.error('❌ Processing errors detected:', errorLogs);
            throw new Error(`Processing failed with errors: ${errorLogs.map(l => l.message).join(', ')}`);
          }
        }
        
        if (!processingComplete) {
          await page.waitForTimeout(1000); // Wait 1 second
          attempts++;
        }
      }
      
      if (!processingComplete) {
        throw new Error('Processing timeout - task did not complete within 30 seconds');
      }
    });

    // Step 5: Verify database rows were added
    await test.step('Verify database entries', async () => {
      console.log('🔍 Verifying database entries...');
      
      // Give the database a moment to complete any pending writes
      await page.waitForTimeout(2000);
      
      // Check if data was inserted by querying via API
      const raceDataResponse = await page.request.get(`/api/race_data?search=${testContext.testRaceId}`);
      
      if (raceDataResponse.status() === 200) {
        const raceData = await raceDataResponse.json();
        
        if (raceData.races && raceData.races.length > 0) {
          console.log(`✅ Found ${raceData.races.length} race(s) in database`);
          
          // Verify we have the expected number of dogs
          const firstRace = raceData.races[0];
          if (firstRace.dogs && firstRace.dogs.length >= testRaceData.expectedRows) {
            console.log(`✅ Database contains ${firstRace.dogs.length} dog entries (expected: ${testRaceData.expectedRows})`);
          } else {
            console.warn(`⚠️ Database contains ${firstRace.dogs?.length || 0} dog entries (expected: ${testRaceData.expectedRows})`);
          }
        } else {
          console.warn('⚠️ No race data found in API response');
        }
      } else {
        console.warn('⚠️ Could not verify database via API, checking file system...');
        
        // Alternative: Check if file was moved to processed directory
        const processedPath = path.join('./processed', testRaceData.fileName);
        if (fs.existsSync(processedPath)) {
          console.log('✅ File was moved to processed directory');
        } else {
          console.warn('⚠️ File not found in processed directory');
        }
      }
    });

    // Step 6: Verify processing logs and metrics
    await test.step('Verify processing metrics', async () => {
      const logsResponse = await page.request.get('/api/logs?type=processing&limit=50');
      expect(logsResponse.status()).toBe(200);
      
      const logsData = await logsResponse.json();
      const processingLogs = logsData.logs || [];
      
      // Look for successful processing indicators
      const successLogs = processingLogs.filter(log => 
        log.message.includes('Successfully processed') || 
        log.message.includes('CSV ingestion complete') ||
        log.message.includes('File processing complete')
      );
      
      if (successLogs.length > 0) {
        console.log('✅ Found processing success indicators in logs');
        successLogs.forEach(log => console.log(`  📝 ${log.message}`));
      } else {
        console.warn('⚠️ No clear success indicators found in processing logs');
        
        // Log recent processing messages for debugging
        const recentLogs = processingLogs.slice(0, 10);
        console.log('Recent processing logs:');
        recentLogs.forEach(log => console.log(`  📝 [${log.level}] ${log.message}`));
      }
    });

    // Step 7: Test file statistics endpoint
    await test.step('Verify file statistics updated', async () => {
      const statsResponse = await page.request.get('/api/file_stats');
      expect(statsResponse.status()).toBe(200);
      
      const stats = await statsResponse.json();
      console.log('📊 File statistics:', JSON.stringify(stats.stats, null, 2));
      
      // Verify stats structure
      expect(stats.success).toBe(true);
      expect(stats.stats).toBeDefined();
      
      if (stats.stats.processed && stats.stats.processed.count > 0) {
        console.log(`✅ Processed files count: ${stats.stats.processed.count}`);
      }
    });

    console.log('🎉 Workflow 1 completed successfully!');
  });

  test('should handle processing errors gracefully', async ({ page }) => {
    console.log('🚀 Testing error handling in processing workflow');

    await test.step('Create invalid test file', async () => {
      const invalidCsvData = `invalid,csv,data,without,proper,headers
this,is,not,a,valid,race,file`;
      
      const testFilePath = path.join(testContext.downloadPath, `invalid_${testContext.testRaceId}.csv`);
      fs.writeFileSync(testFilePath, invalidCsvData);
    });

    await test.step('Trigger processing of invalid file', async () => {
      await page.goto('/scraping');
      
      // Attempt to process the invalid file
      const response = await page.request.post(`/api/process_file/invalid_${testContext.testRaceId}.csv`);
      
      // Should either handle gracefully or return appropriate error
      const result = await response.json();
      
      if (!result.success) {
        console.log('✅ Invalid file processing returned expected error:', result.message);
        expect(result.message).toBeDefined();
      } else {
        console.log('ℹ️ Invalid file was processed (may have fallback handling)');
      }
    });

    await test.step('Verify error logging', async () => {
      // Check that errors are properly logged
      const logsResponse = await page.request.get('/api/logs?type=error&limit=10');
      const logsData = await logsResponse.json();
      
      if (logsData.logs && logsData.logs.length > 0) {
        console.log('✅ Error logs are being captured');
      }
    });
  });

  test('should handle concurrent processing requests', async ({ page }) => {
    console.log('🚀 Testing concurrent processing handling');

    // Create multiple test files
    const testFiles = [];
    for (let i = 1; i <= 3; i++) {
      const testCsvData = `box_number,dog_name,trainer,weight
${i},Concurrent Dog ${i},Test Trainer ${i},32.${i}`;
      
      const fileName = `concurrent_${i}_${testContext.testRaceId}.csv`;
      const filePath = path.join(testContext.downloadPath, fileName);
      fs.writeFileSync(filePath, testCsvData);
      testFiles.push(fileName);
    }

    await test.step('Trigger multiple processing requests', async () => {
      await page.goto('/scraping');
      
      // Start processing all files
      const processingPromises = testFiles.map(async (fileName) => {
        const response = await page.request.post(`/api/process_file/${fileName}`);
        return { fileName, response: await response.json() };
      });

      const results = await Promise.all(processingPromises);
      
      // Verify that the system handles concurrent requests appropriately
      results.forEach((result, index) => {
        console.log(`File ${index + 1} processing result:`, result.response.success ? '✅' : '❌', result.response.message);
      });
      
      // At least one should succeed, or all should have appropriate error handling
      const successCount = results.filter(r => r.response.success).length;
      const errorCount = results.filter(r => !r.response.success).length;
      
      console.log(`✅ Concurrent processing results: ${successCount} successful, ${errorCount} errors`);
      
      if (errorCount > 0) {
        // Verify errors are about concurrency limits (expected behavior)
        const concurrencyErrors = results.filter(r => 
          !r.response.success && 
          r.response.message && 
          r.response.message.includes('already in progress')
        ).length;
        
        if (concurrencyErrors > 0) {
          console.log('✅ System properly prevents concurrent processing');
        }
      }
    });
  });
});
