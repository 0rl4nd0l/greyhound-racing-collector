/// <reference types="cypress" />

describe('Endpoints Menu - Cypress parity', () => {
  beforeEach(() => {
    // Intercepts for the two safe endpoints we will trigger
    cy.intercept('GET', '/api/endpoints').as('endpoints');
    cy.intercept('GET', '/api/model/performance').as('perf');
    cy.intercept('POST', '/api/model/monitoring/drift').as('drift');
  });

  it('renders toolbar and lists endpoints (desktop)', () => {
    cy.viewport(1280, 800);
    cy.visit('/');

    // Ensure feature flag is present
    cy.window().its('ENDPOINTS_MENU_ENABLED').should('equal', true);

    // Wait for menu container or fallback toolbar
    cy.get('.endpoints-menu, #endpoints-menu-toolbar', { timeout: 15000 }).should('exist');

    // Ensure the API returns data
    cy.wait('@endpoints').then((interception) => {
      expect(interception.response?.statusCode).to.eq(200);
      expect(interception.response?.body).to.have.property('endpoints');
      expect(interception.response?.body.endpoints).to.be.an('array');
    });
  });

  it('invokes GET and POST endpoints via dropdown (mobile)', () => {
    cy.viewport(375, 812);
    cy.visit('/');

    // Confirm flag and menu visible
    cy.window().its('ENDPOINTS_MENU_ENABLED').should('equal', true);
    cy.get('.endpoints-menu, #endpoints-menu-toolbar', { timeout: 15000 }).should('exist');

    // GET: /api/model/performance
    // Find a select that contains the option with the given value and select it
    cy.get('select option[value="/api/model/performance"]').then($opt => {
      cy.wrap($opt.parent()).select('/api/model/performance');
    });
    cy.wait('@perf').its('response.statusCode').should('eq', 200);

    // POST: /api/model/monitoring/drift — stub prompt
    cy.window().then((win) => {
      cy.stub(win, 'prompt').returns('{"window":10}');
    });
    cy.get('select option[value="/api/model/monitoring/drift"]').then($opt => {
      cy.wrap($opt.parent()).select('/api/model/monitoring/drift');
    });
    cy.wait('@drift').its('response.statusCode').should('eq', 200);
  });
});
