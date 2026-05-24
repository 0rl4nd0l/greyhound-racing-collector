#!/usr/bin/env node
/**
 * jsdom-based check for GPT Rerank badge rendering in interactive-races.js
 * - Loads interactive-races.js into a jsdom window
 * - Sets window.ENABLE_UI_EXPORTS = true to expose displayPredictionResults
 * - Fires DOMContentLoaded to initialize the module
 * - Calls displayPredictionResults with a synthetic result (gpt_rerank.applied)
 * - Asserts the badge exists and prints PASS/FAIL
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { JSDOM } = require('jsdom');

function log(msg) { process.stdout.write(String(msg) + '\n'); }

async function main() {
  const jsPath = path.resolve(process.cwd(), 'static/js/interactive-races.js');
  const scriptText = fs.readFileSync(jsPath, 'utf-8');

  // Create jsdom window and minimal DOM
  const dom = new JSDOM('<!doctype html><html><head></head><body><main></main></body></html>', {
    url: 'http://localhost/',
    pretendToBeVisual: true,
    runScripts: 'outside-only'
  });

  const { window } = dom;
  const { document } = window;

  // Minimal stubs
  window.ENABLE_UI_EXPORTS = true; // enable export hook in interactive-races.js
  window.fetch = (...args) => Promise.reject(new Error('fetch disabled in jsdom check'));
  window.bootstrap = { Alert: function() {} };
  window.console = console;

  // Evaluate the script in the window context
  const sandbox = { window, document, console: console, Event: window.Event, navigator: window.navigator };
  // Ensure timers available on global
  sandbox.setTimeout = window.setTimeout.bind(window);
  sandbox.clearTimeout = window.clearTimeout.bind(window);
  sandbox.setInterval = window.setInterval.bind(window);
  sandbox.clearInterval = window.clearInterval.bind(window);
  // Ensure window/self/global references
  sandbox.window.window = sandbox.window;
  sandbox.self = sandbox.window;
  sandbox.global = sandbox.window;
  const context = vm.createContext(sandbox);

  // Provide minimal races table body to avoid early return in module init
  const table = document.createElement('table');
  const tbody = document.createElement('tbody');
  tbody.id = 'races-table-body';
  table.appendChild(tbody);
  document.querySelector('main').appendChild(table);

  vm.runInContext(scriptText, context, { filename: 'interactive-races.js' });

  // Fire DOMContentLoaded to trigger module initialization and export
  const evt = new window.Event('DOMContentLoaded');
  document.dispatchEvent(evt);

  if (typeof window.displayPredictionResults !== 'function') {
    log('FAIL: displayPredictionResults not exported');
    process.exit(1);
  }

  // Ensure results container exists
  let container = document.getElementById('prediction-results-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'prediction-results-container';
    container.innerHTML = '<div class="card"><div class="card-header">Prediction Results</div><div class="card-body"><div id="prediction-results-body"></div></div></div>';
    document.querySelector('main').appendChild(container);
  }

  // Call exported function with synthetic result including gpt_rerank
  const results = [{
    success: true,
    predictions: [ { dog_name: 'Alpha', win_prob: 0.58 } ],
    gpt_rerank: { applied: true, alpha: 0.65, tokens_used: 77 },
    race_filename: 'jsdom_sanity.csv',
    predictor_used: 'PredictionPipelineV4'
  }];

  window.displayPredictionResults(results);

  // Validate the badge
  const badge = document.querySelector('span.badge.bg-info');
  if (!badge) {
    log('FAIL: GPT Rerank badge not found');
    process.exit(1);
  }
  const title = badge.getAttribute('title') || '';
  if (!/GPT rerank applied/i.test(title)) {
    log('FAIL: GPT Rerank badge present but tooltip missing');
    process.exit(1);
  }
  if (!/GPT\s*Rerank/i.test(badge.textContent || '')) {
    log('FAIL: GPT Rerank badge text missing');
    process.exit(1);
  }

  log('PASS: GPT Rerank badge rendered with tooltip in jsdom');
  process.exit(0);
}

main().catch((e) => {
  console.error('ERROR:', e);
  process.exit(2);
});

