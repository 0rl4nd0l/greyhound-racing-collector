/*
  Jest test to verify GPT rerank UI changes exist in interactive-races.js (static verification).
  It checks for presence of summary badge and details info card markup.
*/

const fs = require('fs');
const path = require('path');

describe('interactive-races GPT rerank UI (static)', () => {
  let scriptText;

  beforeAll(() => {
    const jsPath = path.resolve(process.cwd(), 'static/js/interactive-races.js');
    scriptText = fs.readFileSync(jsPath, 'utf-8');
  });

  test('contains summary GPT Rerank badge markup', () => {
    expect(scriptText).toContain('badge bg-info ms-2');
    expect(scriptText).toMatch(/GPT\s+Rerank/);
    expect(scriptText).toMatch(/title=\"GPT rerank applied/);
  });

  test('contains details view GPT rerank info card', () => {
    expect(scriptText).toMatch(/GPT rerank applied/);
    expect(scriptText).toMatch(/card-body p-2/);
    expect(scriptText).toMatch(/badge bg-info/);
  });
});

