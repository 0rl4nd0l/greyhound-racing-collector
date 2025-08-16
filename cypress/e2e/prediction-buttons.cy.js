// Cypress tests for Prediction Buttons and UI Flows
// Tests cover single prediction, batch prediction, and UI state management

describe('Prediction Buttons and UI Flows', () => {
  beforeEach(() => {
    // Mock common API responses
    cy.intercept('GET', '/api/system_status', {
      statusCode: 200,
      body: {
        success: true,
        logs: [
          { level: 'INFO', message: 'System running normally' },
          { level: 'WARNING', message: 'Low memory warning' }
        ],
        model_metrics: [
          { model_name: 'V3 Pipeline', accuracy: 0.85 }
        ],
        db_stats: { total_races: 150 }
      }
    }).as('systemStatus');

    cy.intercept('POST', '/api/predict_single_race_enhanced', (req) => {
      const body = req.body;
      const raceId = body.race_id || 'test-race';
      const raceFilename = body.race_filename || 'test-race.csv';
      
      // Simulate successful prediction
      req.reply({
        statusCode: 200,
        body: {
          success: true,
          race_id: raceId,
          race_filename: raceFilename,
          predictions: [
            {
              dog_name: 'Test Dog 1',
              final_score: 0.85,
              box_number: 1
            },
            {
              dog_name: 'Test Dog 2',
              final_score: 0.72,
              box_number: 2
            }
          ],
          predictor_used: 'PredictionPipelineV3',
          message: 'Prediction completed successfully'
        }
      });
    }).as('singlePrediction');

    cy.intercept('POST', '/api/predict_all_upcoming_races_enhanced', {
      statusCode: 200,
      body: {
        success: true,
        total_races: 3,
        success_count: 2,
        predictions: [
          {
            success: true,
            race_filename: 'race1.csv',
            predictions: [
              { dog_name: 'Winner Dog', final_score: 0.9, box_number: 1 }
            ]
          },
          {
            success: true,
            race_filename: 'race2.csv',
            predictions: [
              { dog_name: 'Second Dog', final_score: 0.8, box_number: 3 }
            ]
          }
        ]
      }
    }).as('allPredictions');

    // Visit a test page with prediction buttons
    cy.visit('/test-predictions');
  });

  describe('Single Prediction Button', () => {
    it('should handle single prediction with race ID', () => {
      // Create a test button with race ID
      cy.get('body').then($body => {
        $body.append(`
          <button class="btn btn-primary predict-btn" data-race-id="test-race-123">
            Predict Race
          </button>
          <div id="prediction-results-container" style="display: none;">
            <div id="prediction-results-body"></div>
          </div>
        `);
      });

      // Click the prediction button
      cy.get('.predict-btn').click();

      // Check loading state
      cy.get('.predict-btn').should('contain', 'Predicting...');
      cy.get('.predict-btn').should('have.class', 'btn-secondary');
      cy.get('.predict-btn').should('be.disabled');

      // Wait for API call
      cy.wait('@singlePrediction');

      // Check success state
      cy.get('.predict-btn').should('contain', 'Predicted!');
      cy.get('.predict-btn').should('have.class', 'btn-success');

      // Check results are displayed
      cy.get('#prediction-results-container').should('be.visible');
      cy.get('#prediction-results-body').should('contain', 'Test Dog 1');
      cy.get('#prediction-results-body').should('contain', '85.0%');

      // Check toast notification
      cy.get('.alert-success').should('contain', 'Prediction completed successfully');

      // Wait for button reset
      cy.wait(3500);
      cy.get('.predict-btn').should('contain', 'Predict Race');
      cy.get('.predict-btn').should('have.class', 'btn-primary');
      cy.get('.predict-btn').should('not.be.disabled');
    });

    it('should handle single prediction with race filename', () => {
      // Create a test button with race filename
      cy.get('body').then($body => {
        $body.append(`
          <button class="btn btn-primary predict-btn" data-race-filename="upcoming-race.csv">
            Predict Upcoming Race
          </button>
          <div id="prediction-results-container" style="display: none;">
            <div id="prediction-results-body"></div>
          </div>
        `);
      });

      cy.get('.predict-btn').click();
      cy.wait('@singlePrediction');

      cy.get('.predict-btn').should('contain', 'Predicted!');
      cy.get('#prediction-results-container').should('be.visible');
    });

    it('should handle prediction failure', () => {
      // Mock failed prediction
      cy.intercept('POST', '/api/predict_single_race_enhanced', {
        statusCode: 500,
        body: {
          success: false,
          message: 'Prediction pipeline unavailable'
        }
      }).as('failedPrediction');

      cy.get('body').then($body => {
        $body.append(`
          <button class="btn btn-primary predict-btn" data-race-id="failed-race">
            Predict Race
          </button>
        `);
      });

      cy.get('.predict-btn').click();
      cy.wait('@failedPrediction');

      // Check error state
      cy.get('.predict-btn').should('contain', 'Failed');
      cy.get('.predict-btn').should('have.class', 'btn-danger');

      // Check error toast
      cy.get('.alert-danger').should('contain', 'Prediction failed');
    });

    it('should prevent duplicate predictions', () => {
      cy.get('body').then($body => {
        $body.append(`
          <button class="btn btn-primary predict-btn" data-race-id="duplicate-race">
            Predict Race
          </button>
        `);
      });

      // Click button multiple times quickly
      cy.get('.predict-btn').click();
      cy.get('.predict-btn').click();

      // Should only see one API call
      cy.get('@singlePrediction.all').should('have.length', 1);

      // Should see warning toast
      cy.get('.alert-warning').should('contain', 'already in progress');
    });
  });

  describe('Batch Prediction Button', () => {
    beforeEach(() => {
      // Setup batch prediction UI
      cy.get('body').then($body => {
        $body.append(`
          <input type="checkbox" class="race-checkbox" data-race-id="race1" data-race-filename="race1.csv">
          <input type="checkbox" class="race-checkbox" data-race-id="race2" data-race-filename="race2.csv">
          <input type="checkbox" class="race-checkbox" data-race-id="race3">
          <button class="btn btn-primary run-batch-predictions">
            Run Batch Predictions
          </button>
          <div id="prediction-results-container" style="display: none;">
            <div id="prediction-results-body"></div>
          </div>
        `);
      });
    });

    it('should handle batch prediction with selected races', () => {
      // Select some checkboxes
      cy.get('.race-checkbox').first().check();
      cy.get('.race-checkbox').eq(1).check();

      // Click batch prediction button
      cy.get('.run-batch-predictions').click();

      // Check loading state with progress
      cy.get('.run-batch-predictions').should('contain', 'Predicting 2 races...');
      cy.get('.run-batch-predictions').should('be.disabled');

      // Wait for predictions to complete
      cy.wait('@singlePrediction');
      cy.wait('@singlePrediction');

      // Check completion
      cy.get('.run-batch-predictions').should('contain', 'Completed: 2/2');
      cy.get('.run-batch-predictions').should('have.class', 'btn-success');

      // Check results display
      cy.get('#prediction-results-container').should('be.visible');
      cy.get('#prediction-results-body').should('contain', 'Test Dog 1');

      // Check completion toast
      cy.get('.alert-info').should('contain', 'Batch prediction completed: 2/2 successful');
    });

    it('should warn when no races are selected', () => {
      // Click without selecting any races
      cy.get('.run-batch-predictions').click();

      // Should see warning toast
      cy.get('.alert-warning').should('contain', 'Please select races to predict');

      // Button should not change state
      cy.get('.run-batch-predictions').should('not.be.disabled');
    });

    it('should handle mixed success/failure in batch', () => {
      // Mock one success and one failure
      let callCount = 0;
      cy.intercept('POST', '/api/predict_single_race_enhanced', (req) => {
        callCount++;
        if (callCount === 1) {
          req.reply({
            statusCode: 200,
            body: {
              success: true,
              predictions: [{ dog_name: 'Winner', final_score: 0.9 }]
            }
          });
        } else {
          req.reply({
            statusCode: 500,
            body: {
              success: false,
              message: 'Prediction failed'
            }
          });
        }
      }).as('mixedPredictions');

      cy.get('.race-checkbox').first().check();
      cy.get('.race-checkbox').eq(1).check();
      cy.get('.run-batch-predictions').click();

      cy.wait('@mixedPredictions');
      cy.wait('@mixedPredictions');

      cy.get('.run-batch-predictions').should('contain', 'Completed: 1/2');
      cy.get('.alert-info').should('contain', '1/2 successful');
    });
  });

  describe('Run All Predictions Button', () => {
    it('should handle run all upcoming predictions', () => {
      cy.get('body').then($body => {
        $body.append(`
          <button class="btn btn-primary run-all-predictions">
            Run All Predictions
          </button>
          <div id="prediction-results-container" style="display: none;">
            <div id="prediction-results-body"></div>
          </div>
        `);
      });

      cy.get('.run-all-predictions').click();

      // Check loading state
      cy.get('.run-all-predictions').should('contain', 'Running all predictions...');
      cy.get('.run-all-predictions').should('be.disabled');

      cy.wait('@allPredictions');

      // Check success state
      cy.get('.run-all-predictions').should('contain', 'Completed: 2/3');
      cy.get('.run-all-predictions').should('have.class', 'btn-success');

      // Check results display
      cy.get('#prediction-results-container').should('be.visible');
      cy.get('#prediction-results-body').should('contain', 'Winner Dog');

      // Check success toast
      cy.get('.alert-success').should('contain', 'All predictions completed: 2/3 successful');
    });

    it('should handle run all predictions failure', () => {
      cy.intercept('POST', '/api/predict_all_upcoming_races_enhanced', {
        statusCode: 500,
        body: {
          success: false,
          message: 'No upcoming races found'
        }
      }).as('allPredictionsFailed');

      cy.get('body').then($body => {
        $body.append(`
          <button class="btn btn-primary run-all-predictions">
            Run All Predictions
          </button>
        `);
      });

      cy.get('.run-all-predictions').click();
      cy.wait('@allPredictionsFailed');

      cy.get('.run-all-predictions').should('contain', 'Failed');
      cy.get('.run-all-predictions').should('have.class', 'btn-danger');
      cy.get('.alert-danger').should('contain', 'All predictions failed');
    });
  });

  describe('UI State Management', () => {
    it('should display prediction results correctly', () => {
      const testResult = {
        success: true,
        race_filename: 'test-race.csv',
        predictions: [
          {
            dog_name: 'Champion Dog',
            final_score: 0.92,
            box_number: 1
          }
        ],
        predictor_used: 'PredictionPipelineV3'
      };

      cy.get('body').then($body => {
        $body.append(`
          <div id="prediction-results-container" style="display: none;">
            <div id="prediction-results-body"></div>
          </div>
        `);
      });

      cy.window().then((win) => {
        const manager = new win.PredictionButtonManager();
        manager.displayPredictionResult(testResult);
      });

      cy.get('#prediction-results-container').should('be.visible');
      cy.get('#prediction-results-body').should('contain', 'Champion Dog');
      cy.get('#prediction-results-body').should('contain', '92.0%');
      cy.get('#prediction-results-body').should('contain', 'PredictionPipelineV3');
    });

    it('should handle missing prediction results container gracefully', () => {
      cy.get('body').then($body => {
        $body.append(`
          <button class="btn btn-primary predict-btn" data-race-id="test-race">
            Predict Race
          </button>
        `);
      });

      cy.get('.predict-btn').click();
      cy.wait('@singlePrediction');

      // Should not throw error even without results container
      cy.get('.predict-btn').should('contain', 'Predicted!');
    });

    it('should show appropriate toasts for different scenarios', () => {
      cy.window().then((win) => {
        const manager = new win.PredictionButtonManager();
        
        manager.showSuccessToast('Success message');
        manager.showErrorToast('Error message');
        manager.showWarningToast('Warning message');
        manager.showInfoToast('Info message');
      });

      cy.get('.alert-success').should('contain', 'Success message');
      cy.get('.alert-danger').should('contain', 'Error message');
      cy.get('.alert-warning').should('contain', 'Warning message');
      cy.get('.alert-info').should('contain', 'Info message');

      // Toasts should auto-remove after 5 seconds
      cy.wait(5500);
      cy.get('.alert').should('not.exist');
    });
  });

  describe('Sidebar Polling and EventSource', () => {
    it('should use EventSource when supported', () => {
      // Mock EventSource
      cy.window().then((win) => {
        win.EventSource = class MockEventSource {
          constructor(url) {
            this.url = url;
            this.readyState = 1;
            setTimeout(() => {
              if (this.onmessage) {
                this.onmessage({
                  data: JSON.stringify({
                    type: 'result',
                    logs: [{ level: 'INFO', message: 'EventSource test' }]
                  })
                });
              }
            }, 100);
          }
          close() {}
        };
      });

      cy.visit('/test-sidebar');
      
      // Should receive EventSource updates
      cy.get('#sidebar-logs', { timeout: 1000 }).should('contain', 'EventSource test');
    });

    it('should fallback to polling when EventSource fails', () => {
      // Mock EventSource that fails
      cy.window().then((win) => {
        win.EventSource = class FailingEventSource {
          constructor(url) {
            setTimeout(() => {
              if (this.onerror) {
                this.onerror(new Error('EventSource failed'));
              }
            }, 50);
          }
          close() {}
        };
      });

      cy.visit('/test-sidebar');
      
      // Should fallback to polling and receive system status
      cy.wait('@systemStatus');
      cy.get('#sidebar-logs').should('contain', 'System running normally');
    });

    it('should implement back-off polling strategy', () => {
      // Mock failed API calls to test back-off
      cy.intercept('GET', '/api/system_status', {
        statusCode: 500,
        body: { success: false, message: 'Server error' }
      }).as('failedStatus');

      cy.window().then((win) => {
        // Disable EventSource to force polling
        win.EventSource = undefined;
      });

      cy.visit('/test-sidebar');
      
      // Should attempt multiple calls with increasing intervals
      cy.wait('@failedStatus');
      
      // Check that errors are logged but polling continues
      cy.wait(6000); // Wait for back-off
      cy.get('@failedStatus.all').should('have.length.greaterThan', 1);
    });
  });
});

// Helper commands for setting up test pages
Cypress.Commands.add('createTestPage', (content) => {
  cy.window().then((win) => {
    win.document.body.innerHTML = content;
  });
});

// Custom commands for prediction testing
Cypress.Commands.add('setupPredictionTest', () => {
  cy.createTestPage(`
    <div id="sidebar-logs"></div>
    <div id="sidebar-model-metrics"></div>
    <div id="sidebar-system-health"></div>
    <div id="prediction-results-container" style="display: none;">
      <div id="prediction-results-body"></div>
    </div>
  `);
});

// Before each test in this file, ensure required scripts are loaded
beforeEach(() => {
  cy.visit('/test-blank-page');
  
  // Inject required scripts
  cy.get('head').invoke('append', 
    '<script src="/static/js/prediction-buttons.js"></script>' +
    '<script src="/static/js/sidebar.js"></script>'
  );
  
  cy.setupPredictionTest();
});
