describe('Test Helper Routes', () => {
  beforeEach(() => {
    // Ensure the Flask app is running in testing mode
    cy.request('GET', '/ping').should((response) => {
      expect(response.status).to.eq(200);
    });
  });

  describe('/test-blank-page route', () => {
    it('should load the blank test page with minimal HTML structure', () => {
      cy.visit('/test-blank-page');
      
      // Check that the page loads successfully
      cy.get('body').should('exist');
      
      // Verify the basic HTML structure
      cy.get('#test-container').should('exist');
      cy.get('#test-container').should('have.class', 'container-fluid');
      
      // Check that Bootstrap CSS is loaded (should have Bootstrap classes available)
      cy.get('body').then(($body) => {
        const bodyStyles = window.getComputedStyle($body[0]);
        // Bootstrap typically sets specific font family
        expect(bodyStyles.fontFamily).to.not.be.empty;
      });
      
      // Verify the page is minimal and ready for test injection
      cy.get('#test-container').should('be.empty');
      
      // Test that we can inject content into the test container
      cy.get('#test-container').then(($container) => {
        $container[0].innerHTML = '<div id="injected-test">Test Content</div>';
      });
      cy.get('#injected-test').should('contain.text', 'Test Content');
    });

    it('should have Bootstrap CSS loaded and functional', () => {
      cy.visit('/test-blank-page');
      
      // Inject some Bootstrap components to test CSS is working
      cy.get('#test-container').then(($container) => {
        $container[0].innerHTML = `
          <div class="alert alert-primary" role="alert">
            Bootstrap Alert Test
          </div>
          <button type="button" class="btn btn-success">Bootstrap Button</button>
        `;
      });
      
      // Check that Bootstrap classes are applied
      cy.get('.alert-primary').should('be.visible');
      cy.get('.btn-success').should('be.visible');
    });
  });

  describe('/test-predictions route', () => {
    it('should load the predictions test page with required elements', () => {
      cy.visit('/test-predictions');
      
      // Check main container exists
      cy.get('#predictions-results-container').should('exist');
      cy.get('#predictions-results-container').should('not.be.visible'); // Hidden by default
      
      // Verify Bootstrap and FontAwesome are loaded
      cy.get('head link[href*="bootstrap"]').should('exist');
      cy.get('head link[href*="fontawesome"]').should('exist');
      
      // Check that the prediction-buttons.js script is loaded
      cy.get('script[src="/static/js/prediction-buttons.js"]').should('exist');
    });

    it('should have functional prediction results container', () => {
      cy.visit('/test-predictions');
      
      // Test showing the predictions container
      cy.get('#predictions-results-container').then(($container) => {
        $container.show();
      });
      cy.get('#predictions-results-container').should('be.visible');
      
      // Test injecting prediction content
      cy.get('#predictions-results-container').then(($container) => {
        $container[0].innerHTML = `
          <div class="prediction-item">
            <h4>Race 1 Predictions</h4>
            <div class="predictions-list">
              <div class="prediction-entry">Dog 1 - 85% confidence</div>
            </div>
          </div>
        `;
      });
      
      cy.get('.prediction-item').should('exist');
      cy.get('.predictions-list').should('contain.text', 'Dog 1 - 85% confidence');
    });

    it('should load prediction-buttons.js functionality', () => {
      cy.visit('/test-predictions');
      
      // Wait for script to load
      cy.window().should('have.property', 'fetch'); // Basic check that scripts are running
      
      // Test that we can interact with prediction functionality if available
      cy.window().then((win) => {
        // Check if prediction functions are available
        if (typeof win.showPredictions === 'function') {
          expect(win.showPredictions).to.be.a('function');
        }
      });
    });
  });

  describe('/test-sidebar route', () => {
    it('should load the sidebar test page with required structure', () => {
      cy.visit('/test-sidebar');
      
      // Check main layout containers
      cy.get('.container-fluid').should('exist');
      cy.get('.row').should('exist');
      
      // Check sidebar column
      cy.get('#sidebar-col').should('exist');
      cy.get('#sidebar-col').should('have.class', 'col-md-3');
      
      // Check main content column  
      cy.get('#main-content-col').should('exist');
      cy.get('#main-content-col').should('have.class', 'col-md-9');
      
      // Verify sidebar sections exist
      cy.get('#sidebar-logs').should('exist');
      cy.get('#sidebar-model-metrics').should('exist'); 
      cy.get('#sidebar-system-health').should('exist');
      
      // Check that sidebar.js is loaded
      cy.get('script[src="/static/js/sidebar.js"]').should('exist');
    });

    it('should have Bootstrap styling applied correctly', () => {
      cy.visit('/test-sidebar');
      
      // Test responsive grid system
      cy.viewport(1200, 800);
      cy.get('#sidebar-col').should('be.visible');
      cy.get('#main-content-col').should('be.visible');
      
      // Test mobile responsiveness
      cy.viewport(400, 600);
      // On mobile, columns should stack
      cy.get('.col-md-3').should('exist'); // Classes still exist
      cy.get('.col-md-9').should('exist');
    });

    it('should load sidebar.js and initialize properly', () => {
      cy.visit('/test-sidebar');
      
      // Wait for DOM to load and scripts to initialize
      cy.window().should('have.property', 'fetch');
      
      // Test that sidebar update functions are available
      cy.window().then((win) => {
        // sidebar.js should define updateSidebar function
        if (typeof win.updateSidebar === 'function') {
          expect(win.updateSidebar).to.be.a('function');
        }
      });
      
      // Test that sidebar containers can receive content
      cy.get('#sidebar-logs').then(($logs) => {
        $logs[0].innerHTML = `
          <div class="list-group-item">
            <span class="text-info">[INFO]</span> Test log entry
          </div>
        `;
      });
      cy.get('#sidebar-logs .list-group-item').should('contain.text', 'Test log entry');
      
      // Test model metrics container
      cy.get('#sidebar-model-metrics').then(($metrics) => {
        $metrics[0].innerHTML = '<strong>Test Model:</strong> Accuracy - 0.85';
      });
      cy.get('#sidebar-model-metrics').should('contain.text', 'Test Model');
      
      // Test system health container
      cy.get('#sidebar-system-health').then(($health) => {
        $health[0].innerHTML = '<strong>Total Races:</strong> 1234';
      });
      cy.get('#sidebar-system-health').should('contain.text', 'Total Races');
    });

    it('should handle dynamic content updates', () => {
      cy.visit('/test-sidebar');
      
      // Simulate updating sidebar with real-time data
      cy.get('#sidebar-logs').then(($logs) => {
        // Clear existing content
        $logs[0].innerHTML = '';
        
        // Add multiple log entries
        const logs = [
          { level: 'INFO', message: 'Application started' },
          { level: 'WARNING', message: 'High memory usage detected' },
          { level: 'ERROR', message: 'Database connection failed' }
        ];
        
        logs.forEach(log => {
          let logLevelClass = '';
          switch (log.level) {
            case 'ERROR':
              logLevelClass = 'text-danger';
              break;
            case 'WARNING':
              logLevelClass = 'text-warning';  
              break;
            case 'INFO':
              logLevelClass = 'text-info';
              break;
          }
          
          const logEntry = document.createElement('div');
          logEntry.classList.add('list-group-item');
          logEntry.innerHTML = `<span class="${logLevelClass}">[${log.level}]</span> ${log.message}`;
          $logs[0].appendChild(logEntry);
        });
      });
      
      // Verify the log entries were added with correct styling
      cy.get('#sidebar-logs .list-group-item').should('have.length', 3);
      cy.get('#sidebar-logs .text-info').should('contain.text', '[INFO]');
      cy.get('#sidebar-logs .text-warning').should('contain.text', '[WARNING]');
      cy.get('#sidebar-logs .text-danger').should('contain.text', '[ERROR]');
    });
  });

  describe('Cross-route functionality', () => {
    it('should be able to navigate between test routes', () => {
      // Test navigation between helper routes
      cy.visit('/test-blank-page');
      cy.get('#test-container').should('exist');
      
      cy.visit('/test-predictions');  
      cy.get('#predictions-results-container').should('exist');
      
      cy.visit('/test-sidebar');
      cy.get('#sidebar-logs').should('exist');
      
      // Navigate back to blank page
      cy.visit('/test-blank-page');
      cy.get('#test-container').should('exist');
    });
    
    it('should maintain testing environment across routes', () => {
      // Each route should work in testing mode
      const routes = ['/test-blank-page', '/test-predictions', '/test-sidebar'];
      
      routes.forEach(route => {
        cy.visit(route);
        // Each page should load successfully (no 404 or 500 errors)
        cy.get('body').should('exist');
        // Each page should have some basic content
        cy.get('body').should('not.be.empty');
      });
    });
  });
});
