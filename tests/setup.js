/**
 * Jest Test Setup
 * Configures testing environment for the Greyhound Racing Predictor
 */

import '@testing-library/jest-dom';
import nock from 'nock';

// Global test configuration
beforeEach(() => {
  // Enforce mocked OpenAI by default in Node tests
  if (!process.env.OPENAI_USE_LIVE) process.env.OPENAI_USE_LIVE = '0';

  // If not using live, set up nock mocks
  if (process.env.OPENAI_USE_LIVE !== '1') {
    nock.disableNetConnect();
    // Allow localhost for Playwright/Jest requests
    nock.enableNetConnect((host) => /(^|\.)localhost(:\d+)?$/.test(host) || /^127\.0\.0\.1(:\d+)?$/.test(host));

    // Mock OpenAI Responses API
    nock('https://api.openai.com')
      .persist()
      .post('/v1/responses')
      .reply(200, { output_text: 'mocked', usage: { total_tokens: 1 } });

    // Mock OpenAI Chat Completions API
    nock('https://api.openai.com')
      .persist()
      .post('/v1/chat/completions')
      .reply(200, { choices: [{ message: { content: 'mocked' } }], usage: { total_tokens: 1 } });
  }
  
  // Mock localStorage
  const localStorageMock = {
    getItem: jest.fn(),
    setItem: jest.fn(),
    removeItem: jest.fn(),
    clear: jest.fn(),
  };
  global.localStorage = localStorageMock;
  
  // Mock sessionStorage
  const sessionStorageMock = {
    getItem: jest.fn(),
    setItem: jest.fn(),
    removeItem: jest.fn(),
    clear: jest.fn(),
  };
  global.sessionStorage = sessionStorageMock;
  
  // Mock console methods to reduce noise in tests
  global.console = {
    ...console,
    // Uncomment the line below to suppress console.log in tests
    // log: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  };

  // Mock Bootstrap tooltips
  global.bootstrap = {
    Tooltip: jest.fn().mockImplementation(() => ({
      show: jest.fn(),
      hide: jest.fn(),
      dispose: jest.fn(),
    })),
  };

  // Mock HTML elements commonly used in the app
  document.body.innerHTML = '';
  
  // Add meta viewport for responsive tests
  const metaViewport = document.createElement('meta');
  metaViewport.name = 'viewport';
  metaViewport.content = 'width=device-width, initial-scale=1';
  document.head.appendChild(metaViewport);
});

afterEach(() => {
  // Clean up DOM after each test
  document.body.innerHTML = '';
  document.head.innerHTML = '';
  
  // Clear all mocks
  jest.clearAllMocks();

  // Clean nock unless live
  if (process.env.OPENAI_USE_LIVE !== '1') {
    nock.cleanAll();
    nock.enableNetConnect();
  }
});

// Custom matchers for better assertions
expect.extend({
  toBeValidPrediction(received) {
    const pass = 
      typeof received === 'object' &&
      received !== null &&
      typeof received.win_probability === 'number' &&
      received.win_probability >= 0 &&
      received.win_probability <= 1 &&
      typeof received.place_probability === 'number' &&
      received.place_probability >= 0 &&
      received.place_probability <= 1;

    if (pass) {
      return {
        message: () => `expected ${JSON.stringify(received)} not to be a valid prediction`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${JSON.stringify(received)} to be a valid prediction with win_probability and place_probability between 0 and 1`,
        pass: false,
      };
    }
  },
  
  toHaveValidRaceStructure(received) {
    const pass = 
      typeof received === 'object' &&
      received !== null &&
      typeof received.race_id === 'string' &&
      typeof received.venue === 'string' &&
      typeof received.race_number === 'number' &&
      Array.isArray(received.runners);

    if (pass) {
      return {
        message: () => `expected ${JSON.stringify(received)} not to have valid race structure`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${JSON.stringify(received)} to have valid race structure with race_id, venue, race_number, and runners array`,
        pass: false,
      };
    }
  }
});

// Mock API responses for common endpoints
export const mockApiResponses = {
  races: {
    success: true,
    races: [
      {
        race_id: 'test-race-1',
        venue: 'Test Venue',
        race_number: 1,
        race_date: '2024-01-15',
        race_name: 'Test Race',
        grade: 'Grade 5',
        distance: 500,
        runners: [
          {
            dog_name: 'Test Dog 1',
            box_number: 1,
            odds: '2.50',
            win_probability: 0.4,
            place_probability: 0.7,
            confidence: 0.8
          },
          {
            dog_name: 'Test Dog 2',
            box_number: 2,
            odds: '3.00',
            win_probability: 0.3,
            place_probability: 0.6,
            confidence: 0.7
          }
        ]
      }
    ],
    pagination: {
      page: 1,
      per_page: 10,
      total_count: 1,
      total_pages: 1,
      has_next: false,
      has_prev: false
    }
  },
  
  predictions: {
    success: true,
    predictions: [
      {
        race_id: 'test-race-1',
        predictions: {
          'Test Dog 1': { win_prob: 0.4, place_prob: 0.7 },
          'Test Dog 2': { win_prob: 0.3, place_prob: 0.6 }
        }
      }
    ]
  },
  
  dogs: {
    success: true,
    dogs: [
      {
        dog_name: 'Test Dog 1',
        total_races: 10,
        total_wins: 4,
        win_percentage: 40.0,
        place_percentage: 70.0
      }
    ]
  }
};
