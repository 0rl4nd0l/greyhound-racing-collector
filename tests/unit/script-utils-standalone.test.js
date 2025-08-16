/**
 * Unit Tests for script.js utility functions - Standalone Version
 * Tests individual utility functions with custom DOM mocking
 */

import fs from 'fs';
import path from 'path';

// Load sample API responses
const loadSampleData = (filename) => {
  const filePath = path.join(__dirname, '../data/api_responses', filename);
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
};

// Utility functions extracted from script.js for testing
function formatFileSize(bytes) {
  if (bytes === null || bytes === undefined || bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  
  // Handle small positive and negative numbers
  if (Math.abs(bytes) < k) {
    return parseFloat(bytes.toFixed(2)) + ' Bytes';
  }
  
  const i = Math.floor(Math.log(Math.abs(bytes)) / Math.log(k));
  
  // Ensure we don't go beyond our sizes array
  const sizeIndex = Math.min(i, sizes.length - 1);
  
  return parseFloat((bytes / Math.pow(k, sizeIndex)).toFixed(2)) + ' ' + sizes[sizeIndex];
}

function formatDate(dateString) {
  const date = new Date(dateString);
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function showNotification(message, type = 'info') {
  const notification = mockDocument.createElement('div');
  notification.className = `notification notification-${type}`;
  notification.textContent = message;
  
  mockDocument.body.appendChild(notification);
  
  setTimeout(() => {
    notification.remove();
  }, 3000);
}

// Mock DOM objects
let mockDocument = {};

// Dashboard utility functions
const dashboardUtils = {
  async refreshStats() {
    try {
      const response = await fetch('/api/stats');
      const data = await response.json();
      console.log('📊 Stats refreshed:', data);
      return data;
    } catch (error) {
      console.error('Error refreshing stats:', error);
      throw error;
    }
  },

  refreshSection(sectionId) {
    const section = mockDocument.getElementById(sectionId);
    if (section) {
      section.style.opacity = '0.5';
      
      setTimeout(() => {
        section.style.opacity = '1';
      }, 500);
    }
  },

  async fetchRecentRaces(limit = 10) {
    try {
      const response = await fetch(`/api/recent_races?limit=${limit}`);
      const data = await response.json();
      return data.races || [];
    } catch (error) {
      console.error('Error fetching recent races:', error);
      return [];
    }
  },

  async fetchRaceDetails(raceId) {
    try {
      const response = await fetch(`/api/race/${raceId}`);
      const data = await response.json();
      return data.race_data;
    } catch (error) {
      console.error('Error fetching race details:', error);
      return null;
    }
  }
};

// Custom matchers for our tests
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

// Setup for each test
beforeEach(() => {
  // Mock DOM elements
  const mockElement = {
    className: '',
    textContent: '',
    remove: jest.fn(),
    style: { opacity: '1' },
    setAttribute: jest.fn(),
    getAttribute: jest.fn()
  };

  mockDocument = {
    createElement: jest.fn(() => ({ ...mockElement })),
    getElementById: jest.fn(() => ({ ...mockElement })),
    body: { appendChild: jest.fn() },
    head: { appendChild: jest.fn() },
    querySelectorAll: jest.fn(() => [])
  };

  // Mock globals
  global.fetch = jest.fn();
  global.console = { ...console, log: jest.fn(), error: jest.fn() };
  global.setTimeout = jest.fn((fn) => {
    if (typeof fn === 'function') fn();
    return 123;
  });
});

describe('formatFileSize', () => {
  test('formats zero bytes correctly', () => {
    expect(formatFileSize(0)).toBe('0 Bytes');
  });

  test('formats bytes correctly', () => {
    expect(formatFileSize(500)).toBe('500 Bytes');
    expect(formatFileSize(1023)).toBe('1023 Bytes');
  });

  test('formats KB correctly', () => {
    expect(formatFileSize(1024)).toBe('1 KB');
    expect(formatFileSize(1536)).toBe('1.5 KB');
    expect(formatFileSize(2048)).toBe('2 KB');
  });

  test('formats MB correctly', () => {
    expect(formatFileSize(1048576)).toBe('1 MB');
    expect(formatFileSize(1572864)).toBe('1.5 MB');
    expect(formatFileSize(5242880)).toBe('5 MB');
  });

  test('formats GB correctly', () => {
    expect(formatFileSize(1073741824)).toBe('1 GB');
    expect(formatFileSize(2147483648)).toBe('2 GB');
    expect(formatFileSize(5368709120)).toBe('5 GB');
  });

  test('handles decimal precision correctly', () => {
    expect(formatFileSize(1536000)).toBe('1.46 MB');
    expect(formatFileSize(1234567)).toBe('1.18 MB');
  });

  test('handles very large numbers', () => {
    expect(formatFileSize(1099511627776)).toBe('1 TB');
  });

  test('handles sample API response storage sizes', () => {
    const statsData = loadSampleData('stats.json');
    const storageUsed = statsData.stats.database_metrics.storage_used;
    
    const formattedSize = formatFileSize(storageUsed);
    expect(formattedSize).toMatch(/^\d+(\.\d+)? (KB|MB|GB)$/);
    expect(formatFileSize(4847295738)).toBe('4.51 GB');
  });

  test('handles negative file sizes', () => {
    expect(formatFileSize(-1024)).toBe('-1 KB');
    expect(formatFileSize(-1)).toBe('-1 Bytes');
  });

  test('handles very small numbers', () => {
    expect(formatFileSize(0.5)).toBe('0.5 Bytes');
    expect(formatFileSize(0.1)).toBe('0.1 Bytes');
  });
});

describe('formatDate', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2024-01-15T10:00:00Z'));
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('formats ISO date string correctly', () => {
    const result = formatDate('2024-01-15T14:30:00Z');
    expect(result).toMatch(/\d+\/\d+\/\d+/); // Date part
    expect(result).toMatch(/\d+:\d+:\d+/);    // Time part
  });

  test('formats regular date string', () => {
    const result = formatDate('2024-01-15');
    expect(result).toMatch(/\d+\/\d+\/\d+/);
  });

  test('handles sample API response dates', () => {
    const racesData = loadSampleData('races.json');
    const raceDate = racesData.races[0].race_date;
    
    const formattedDate = formatDate(raceDate);
    expect(formattedDate).toMatch(/\d+\/\d+\/\d+/);
    expect(formattedDate).toMatch(/\d+:\d+:\d+/);
  });

  test('handles prediction timestamp', () => {
    const predictionsData = loadSampleData('predictions.json');
    const timestamp = predictionsData.predictions[0].prediction_timestamp;
    
    const formattedDate = formatDate(timestamp);
    expect(formattedDate).toBeDefined();
    expect(typeof formattedDate).toBe('string');
  });

  test('handles stats timestamps', () => {
    const statsData = loadSampleData('stats.json');
    const lastUpdate = statsData.stats.system_metrics.last_update;
    
    const formattedDate = formatDate(lastUpdate);
    expect(formattedDate).toBeDefined();
    expect(typeof formattedDate).toBe('string');
  });

  test('handles invalid date gracefully', () => {
    const result = formatDate('invalid-date');
    expect(result).toMatch(/Invalid Date|NaN/);
  });
});

describe('showNotification', () => {
  test('creates notification with default type', () => {
    showNotification('Test message');
    
    expect(mockDocument.createElement).toHaveBeenCalledWith('div');
    expect(mockDocument.body.appendChild).toHaveBeenCalled();
  });

  test('creates notification with custom type', () => {
    showNotification('Error message', 'error');
    expect(mockDocument.body.appendChild).toHaveBeenCalled();
  });

  test('removes notification after timeout', () => {
    const mockElement = { remove: jest.fn() };
    mockDocument.createElement = jest.fn(() => mockElement);
    
    showNotification('Test message');
    
    expect(setTimeout).toHaveBeenCalledWith(expect.any(Function), 3000);
    expect(mockElement.remove).toHaveBeenCalled();
  });
});

describe('dashboardUtils.refreshStats', () => {
  test('makes fetch request to stats endpoint', async () => {
    const mockResponse = loadSampleData('stats.json');
    global.fetch.mockResolvedValue({
      json: () => Promise.resolve(mockResponse)
    });

    const result = await dashboardUtils.refreshStats();

    expect(fetch).toHaveBeenCalledWith('/api/stats');
    expect(result).toEqual(mockResponse);
    expect(console.log).toHaveBeenCalledWith('📊 Stats refreshed:', mockResponse);
  });

  test('handles fetch error gracefully', async () => {
    const error = new Error('Network error');
    global.fetch.mockRejectedValue(error);

    await expect(dashboardUtils.refreshStats()).rejects.toThrow('Network error');
    expect(console.error).toHaveBeenCalledWith('Error refreshing stats:', error);
  });

  test('processes real stats data structure', async () => {
    const statsData = loadSampleData('stats.json');
    global.fetch.mockResolvedValue({
      json: () => Promise.resolve(statsData)
    });

    const result = await dashboardUtils.refreshStats();

    expect(fetch).toHaveBeenCalled();
    expect(result).toEqual(expect.objectContaining({
      success: true,
      stats: expect.objectContaining({
        system_metrics: expect.any(Object),
        recent_performance: expect.any(Object)
      })
    }));
  });
});

describe('dashboardUtils.refreshSection', () => {
  test('calls getElementById and sets up setTimeout', () => {
    const mockSection = { style: { opacity: '1' } };
    mockDocument.getElementById = jest.fn(() => mockSection);

    dashboardUtils.refreshSection('test-section');

    expect(mockDocument.getElementById).toHaveBeenCalledWith('test-section');
    expect(setTimeout).toHaveBeenCalledWith(expect.any(Function), 500);
    // Since our mock setTimeout executes immediately, opacity should be restored to 1
    expect(mockSection.style.opacity).toBe('1');
  });

  test('modifies section opacity during refresh flow', () => {
    const mockSection = { style: { opacity: '1' } };
    mockDocument.getElementById = jest.fn(() => mockSection);
    
    // Mock setTimeout to not execute immediately so we can test the intermediate state
    global.setTimeout = jest.fn();

    dashboardUtils.refreshSection('test-section');

    expect(mockDocument.getElementById).toHaveBeenCalledWith('test-section');
    expect(mockSection.style.opacity).toBe('0.5'); // Should be set to 0.5 first
    expect(setTimeout).toHaveBeenCalledWith(expect.any(Function), 500);
  });

  test('handles non-existent section gracefully', () => {
    mockDocument.getElementById = jest.fn(() => null);

    expect(() => {
      dashboardUtils.refreshSection('non-existent');
    }).not.toThrow();
  });
});

describe('dashboardUtils.fetchRecentRaces', () => {
  test('fetches recent races with default limit', async () => {
    const mockResponse = loadSampleData('races.json');
    global.fetch.mockResolvedValue({
      json: () => Promise.resolve(mockResponse)
    });

    const result = await dashboardUtils.fetchRecentRaces();

    expect(fetch).toHaveBeenCalledWith('/api/recent_races?limit=10');
    expect(result).toEqual(mockResponse.races);
  });

  test('fetches recent races with custom limit', async () => {
    const mockResponse = loadSampleData('races.json');
    global.fetch.mockResolvedValue({
      json: () => Promise.resolve(mockResponse)
    });

    const result = await dashboardUtils.fetchRecentRaces(5);

    expect(fetch).toHaveBeenCalledWith('/api/recent_races?limit=5');
    expect(result).toEqual(mockResponse.races);
  });

  test('handles fetch error and returns empty array', async () => {
    global.fetch.mockRejectedValue(new Error('Network error'));

    const result = await dashboardUtils.fetchRecentRaces();

    expect(result).toEqual([]);
    expect(console.error).toHaveBeenCalledWith('Error fetching recent races:', expect.any(Error));
  });

  test('processes real race data structure', async () => {
    const racesData = loadSampleData('races.json');
    global.fetch.mockResolvedValue({
      json: () => Promise.resolve(racesData)
    });

    const result = await dashboardUtils.fetchRecentRaces();

    expect(result).toBeInstanceOf(Array);
    expect(result[0]).toHaveValidRaceStructure();
    expect(result[0].runners).toBeInstanceOf(Array);
    expect(result[0].runners[0]).toHaveProperty('dog_name');
    expect(result[0].runners[0]).toHaveProperty('win_probability');
  });

  test('handles missing races property', async () => {
    global.fetch.mockResolvedValue({
      json: () => Promise.resolve({ success: true, data: 'no races property' })
    });

    const result = await dashboardUtils.fetchRecentRaces();

    expect(result).toEqual([]);
  });
});

describe('dashboardUtils.fetchRaceDetails', () => {
  test('fetches race details by ID', async () => {
    const mockResponse = loadSampleData('race_details.json');
    global.fetch.mockResolvedValue({
      json: () => Promise.resolve(mockResponse)
    });

    const result = await dashboardUtils.fetchRaceDetails('race-001');

    expect(fetch).toHaveBeenCalledWith('/api/race/race-001');
    expect(result).toEqual(mockResponse.race_data);
  });

  test('handles fetch error and returns null', async () => {
    global.fetch.mockRejectedValue(new Error('Race not found'));

    const result = await dashboardUtils.fetchRaceDetails('invalid-race');

    expect(result).toBeNull();
    expect(console.error).toHaveBeenCalledWith('Error fetching race details:', expect.any(Error));
  });

  test('processes detailed race data structure', async () => {
    const raceDetailsData = loadSampleData('race_details.json');
    global.fetch.mockResolvedValue({
      json: () => Promise.resolve(raceDetailsData)
    });

    const result = await dashboardUtils.fetchRaceDetails('race-001-melb-2024-01-15');

    expect(result).toHaveProperty('race_id');
    expect(result).toHaveProperty('venue');
    expect(result).toHaveProperty('runners');
    expect(result.runners[0]).toHaveProperty('form_data');
    expect(result.runners[0]).toHaveProperty('predictions');
    expect(result.runners[0].predictions).toBeValidPrediction();
  });
});

describe('DOM Integration Tests', () => {
  test('tooltip re-initialization does not throw errors', () => {
    // Mock Bootstrap tooltip initialization
    global.bootstrap = {
      Tooltip: jest.fn().mockImplementation(() => ({
        show: jest.fn(),
        hide: jest.fn(),
        dispose: jest.fn()
      }))
    };

    // Mock elements with data-bs-toggle
    const mockElements = [
      { setAttribute: jest.fn(), getAttribute: jest.fn() },
      { setAttribute: jest.fn(), getAttribute: jest.fn() }
    ];
    
    mockDocument.querySelectorAll = jest.fn((selector) => {
      if (selector === '[data-bs-toggle="tooltip"]') {
        return mockElements;
      }
      return [];
    });

    // Test tooltip re-initialization
    expect(() => {
      mockElements.forEach(el => {
        new global.bootstrap.Tooltip(el);
      });
    }).not.toThrow();

    expect(global.bootstrap.Tooltip).toHaveBeenCalledTimes(2);
  });
});

describe('Snapshot Tests', () => {
  test('formatFileSize output snapshots', () => {
    const testSizes = [0, 512, 1024, 1536, 1048576, 5242880, 1073741824];
    const results = testSizes.map(size => ({
      input: size,
      output: formatFileSize(size)
    }));
    
    expect(results).toMatchSnapshot('file-size-formatting');
  });

  test('API response processing snapshots', () => {
    const racesData = loadSampleData('races.json');
    const predictionsData = loadSampleData('predictions.json');
    const statsData = loadSampleData('stats.json');
    
    const processed = {
      racesCount: racesData.races.length,
      firstRaceRunners: racesData.races[0].runners.length,
      predictionsCount: predictionsData.predictions.length,
      statsAccuracy: statsData.stats.system_metrics.accuracy_rate,
      storageFormatted: formatFileSize(statsData.stats.database_metrics.storage_used)
    };
    
    expect(processed).toMatchSnapshot('api-response-processing');
  });
});

describe('Error Handling and Edge Cases', () => {
  test('handles undefined and null inputs gracefully', () => {
    expect(() => formatFileSize(null)).not.toThrow();
    expect(() => formatFileSize(undefined)).not.toThrow();
    expect(() => formatDate(null)).not.toThrow();
    expect(() => formatDate(undefined)).not.toThrow();
    expect(() => showNotification(null)).not.toThrow();
    expect(() => showNotification(undefined)).not.toThrow();
  });

  test('handles API response with missing data gracefully', async () => {
    // Test with incomplete response
    global.fetch = jest.fn().mockResolvedValue({
      json: () => Promise.resolve({ success: false })
    });

    const result = await dashboardUtils.fetchRecentRaces();
    expect(result).toEqual([]); // Should handle gracefully
  });
});
