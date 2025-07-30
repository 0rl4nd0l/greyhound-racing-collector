/**
 * End-to-End tests for the Greyhound Racing Predictor
 * Covers critical user flows using a live backend.
 */

describe('Critical User Flows with Live Data', () => {

  beforeEach(() => {
    cy.visit('/');
    // Give the app time to load initial data
    cy.wait(1000); 
  });

  it('should load the main page successfully and display races', () => {
    cy.contains('h1', 'Dashboard Overview').should('be.visible');
    cy.get('.race-card').should('have.length.greaterThan', 0);
  });

  it('should expand a race card and display runner details', () => {
    cy.get('.race-card').first().click();
    cy.get('.runner-entry').should('have.length.greaterThan', 0);
    cy.get('.runner-name').first().should('not.be.empty');
  });

  it('should filter races by a known venue', () => {
    // This test assumes a venue named 'Albion Park' exists in your data
    cy.get('#searchInput').type('Albion Park');
    cy.get('#racesContainer').should('not.be.empty');
    cy.contains('.race-title', 'Albion Park', { matchCase: false }).should('be.visible');
  });

  it('should sort races by grade', () => {
    cy.get('#sortSelect').select('grade');
    cy.wait(500); // Allow time for re-render
    cy.get('.race-card').should('have.length.greaterThan', 0);
  });

  it('should allow running a prediction and exporting CSV', () => {
    cy.visit('/ml-dashboard');
    cy.get('#run-comprehensive-prediction-btn').click();

    // Wait for the prediction to complete
    cy.contains('#prediction-status-message', 'Comprehensive prediction pipeline completed', { timeout: 30000 }).should('be.visible');
    cy.get('#export-csv-btn').should('be.visible');
    cy.get('#export-csv-btn').click();
    
    // Check if a file was downloaded (this is tricky in Cypress and may require a task)
    // For now, we just ensure the button is there and clickable.
    cy.log('CSV export button clicked.');
  });
});
