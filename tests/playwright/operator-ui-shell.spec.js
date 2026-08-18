const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

const route = '/operator-ui/prototype';

function computedTimesInMilliseconds(value) {
  return value.split(',').map((duration) => {
    const match = duration.trim().match(/^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(ms|s)$/i);
    expect(match, `valid computed time: ${duration}`).not.toBeNull();
    const milliseconds = Number(match[1]) * (match[2].toLowerCase() === 's' ? 1000 : 1);
    expect(Number.isFinite(milliseconds), `finite computed time: ${duration}`).toBe(true);
    return milliseconds;
  });
}

async function expectNoOverflow(page) {
  const width = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    client: document.documentElement.clientWidth,
  }));
  expect(width.scroll).toBeLessThanOrEqual(width.client);
}

test.describe('atomic fixture-only operator workflow', () => {
  test('golden flow selects, confirms, reconnects, and exposes evidence', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto(route);
    await expect(page.getByText('RESEARCH ONLY — NOT FOR BETTING').first()).toBeVisible();
    await page.getByRole('link', { name: 'Choose an exact race' }).click();
    await expect(page.getByRole('heading', { name: 'Date → meeting → race' })).toBeVisible();
    await expect(page.getByText('1 Apr 2099, 8:30 pm AEST')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Review research-only confirmation' })).toBeEnabled();
    await page.getByRole('button', { name: 'Review research-only confirmation' }).click();
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText('RESEARCH ONLY — NOT FOR BETTING')).toBeVisible();
    await page.keyboard.press('Tab');
    await page.getByRole('button', { name: 'Confirm fixture lifecycle' }).click();
    await expect(page.getByRole('heading', { name: 'One request, one attempt, no retry' })).toBeFocused();
    await expect(page.getByText('FIXTURE-JOB-20990401-0006')).toBeVisible();
    await page.reload();
    await expect(page.getByText('FIXTURE-JOB-20990401-0006')).toBeVisible();
    await page.getByRole('link', { name: 'View ranked probabilities' }).click();
    await expect(page.getByRole('heading', { name: 'Ranked win probabilities' })).toBeVisible();
    await expect(page.getByText('bundle-fixture-20990401-r06')).toBeVisible();
    await expectNoOverflow(page);
    await page.locator('#audit').scrollIntoViewIfNeeded();
    await expect(page.locator('.persistent-labels .research-warning')).toBeInViewport();
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  });

  test('all invalid selection fixtures fail closed with exact distinctions', async ({ page }) => {
    await page.goto(route);
    const selector = page.locator('#fixture-state');
    const review = page.getByRole('button', { name: 'Review research-only confirmation' });
    const fixtures = {
      ambiguous: 'ambiguous race identity',
      'post-jump': 'scheduled jump has passed',
      'missing-runner': 'active runner identity is missing',
      stale: '300-second pre-jump freshness policy',
      'missing-jump': 'scheduled jump identity is missing',
      unsupported: 'window or model configuration is unsupported',
      conflicting: 'source and selected-race evidence conflict',
      unavailable: 'source evidence is unavailable',
    };
    for (const [value, reason] of Object.entries(fixtures)) {
      await selector.selectOption(value);
      await expect(review).toBeDisabled();
      await expect(page.getByRole('status')).toContainText(reason);
    }
    await selector.selectOption('valid');
    await expect(review).toBeEnabled();
  });

  test('mobile contains long identities and preserves navigation and actions', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto(route);
    await expect(page.getByRole('navigation', { name: 'Console sections' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Review research-only confirmation' })).toBeVisible();
    await page.getByText('90766b65ba7f184d53b57c520fd9af1962797c9370984769d93eecc631716cea').scrollIntoViewIfNeeded();
    await expectNoOverflow(page);
    await page.locator('#audit').scrollIntoViewIfNeeded();
    await expect(page.locator('.persistent-labels .research-warning')).toBeInViewport();
  });

  test('keyboard dialog focus and print evidence view are supported', async ({ page }) => {
    await page.goto(route);
    await page.keyboard.press('Tab');
    await expect(page.getByRole('link', { name: 'Skip to prototype content' })).toBeFocused();
    await page.getByRole('button', { name: 'Review research-only confirmation' }).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('dialog')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).toBeHidden();
    await page.emulateMedia({ media: 'print' });
    await expect(page.locator('.sidebar')).toHaveCSS('display', 'none');
    await expect(page.locator('.evidence-view').first()).toBeVisible();
  });

  test('reduced motion preference disables smooth scrolling and animation', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto(route);
    expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true);
    const motion = await page.locator('html').evaluate((element) => {
      const panel = document.querySelector('.panel');
      return {
        scrollBehavior: getComputedStyle(element).scrollBehavior,
        animationDuration: getComputedStyle(panel).animationDuration,
        transitionDuration: getComputedStyle(panel).transitionDuration,
      };
    });

    expect(motion.scrollBehavior).toBe('auto');
    for (const duration of [
      ...computedTimesInMilliseconds(motion.animationDuration),
      ...computedTimesInMilliseconds(motion.transitionDuration),
    ]) {
      expect(duration).toBeLessThanOrEqual(0.01);
    }
  });
});

test.describe('connected read-only operator workflow', () => {
  const envelope = { source_kind:'fixture_api', source_identity:'mock.source.exact', content_sha256:'a'.repeat(64), source_locator:'server.configured.mock', source_at:'2026-07-31T01:02:00Z', generated_at:null, observed_at:null, server_observed_at:'2026-07-31T01:02:03Z', age_seconds:3, freshness_policy:'P-OPS-5', availability:'present', schema_integrity:'valid', reference_hashes:{manifest:'b'.repeat(64)}, evidence_identity:{exact:'mock-1'}, status:'AVAILABLE/FRESH', supported_claim:'Exact mocked read-only browser evidence.' };
  async function connectedPage(page, overrides={}) {
    const sections=['overview','upcoming-races','recent-predictions','collector','corpus','models','system','audit','detail'].map(name=>`<section id="${name}" ${name==='detail'?'hidden':''}><h2 id="${name}-title" tabindex="-1">${name}</h2><article class="panel resource-panel" data-resource="${name}" aria-busy="true"><p class="resource-state">Loading…</p><div class="resource-data"></div><div class="resource-detail"></div></article></section>`).join('');
    const prediction=`<section id="manual-prediction" hidden aria-hidden="true"><form id="prediction-form"><select id="prediction-race"></select><select id="prediction-model"></select><select id="prediction-config"></select><select id="prediction-odds"></select><p id="runner-confirmation"></p><button id="prediction-submit" type="submit"></button></form><div id="job-status" tabindex="-1"></div><ol id="job-timeline"></ol><section id="job-result"></section><details id="job-evidence"><div class="evidence-view"></div></details></section>`;
    const html=`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Greyhound Operator Console — read only</title><link rel="stylesheet" href="/static/css/operator-ui.css"><script src="/static/js/operator-ui-state.js" defer></script><script src="/static/js/operator-ui-connected.js" defer></script></head><body><a class="skip-link" href="#connected-content">Skip to connected content</a><div class="workspace"><header class="topbar"><h1>Operator console</h1><strong>CONNECTED MODE</strong><strong class="research-warning">RESEARCH ONLY — NOT FOR BETTING</strong></header><main id="connected-content"><div id="connection-state" role="status" aria-live="polite">Loading</div>${sections}${prediction}</main></div></body></html>`;
    await page.route('**/operator-ui', request=>request.fulfill({contentType:'text/html',body:html}));
    await page.route('**/operator-ui/api/v1/**', async route=>{
      const request=route.request(); expect(request.method()).toBe('GET'); expect(request.postData()).toBeNull(); expect(new URL(request.url()).search).toBe('');
      const path=new URL(request.url()).pathname; if(overrides[path]==='OFFLINE') return route.abort('failed');
      if(path.endsWith('/r3-capability')) return route.fulfill({contentType:'application/json',body:JSON.stringify({schema:'operator_ui_r3_capability_v1',authorized:false,runtime_configured:false,level:2})});
      const configured=(typeof overrides[path]==='function'?await overrides[path]():overrides[path])||{classification:'AVAILABLE/FRESH',data:{}};
      const resource=path.endsWith('/races/upcoming')?'upcoming_races':path.includes('/races/')?'race_detail':path.endsWith('/predictions/recent')?'recent_predictions':path.includes('/predictions/')?'prediction_detail':path.split('/').at(-1).replaceAll('-','_');
      const payload={schema:'operator_ui_level_1_api_v1',api_version:'v1',resource,classification:configured.classification,stale:configured.classification==='STALE',server_observed_at:configured.server_observed_at||envelope.server_observed_at,evidence:configured.evidence||envelope};
      if(Object.hasOwn(configured,'reason'))payload.reason=configured.reason;else payload.data=configured.data||{};
      await route.fulfill({contentType:'application/json',body:JSON.stringify(payload)});
    });
    await page.goto('/operator-ui');
  }
  test('exact keyboard detail, reload, desktop containment, reduced motion and print evidence', async ({page})=>{
    await page.emulateMedia({reducedMotion:'reduce'}); await page.setViewportSize({width:1280,height:900});
    const available=data=>({classification:'AVAILABLE/FRESH',server_observed_at:envelope.server_observed_at,evidence:envelope,data});
    await connectedPage(page,{'/operator-ui/api/v1/races/upcoming':available({races:[{route_id:'race-route-1',race_id:'exact-race-1'}]}),'/operator-ui/api/v1/races/race-route-1':available({race:{race_id:'exact-race-1',runners:[{box:1,name:'Exact Runner'}]}})});
    const detail=page.getByRole('button',{name:/View exact race detail/}); await detail.focus(); await page.keyboard.press('Enter'); await expect(page.locator('#detail-title')).toBeFocused(); await expect(page.locator('#detail')).toContainText('Exact Runner');
    await page.reload(); await expect(page.getByText('exact-race-1').first()).toBeVisible(); await expectNoOverflow(page);
    expect(await page.locator('html').evaluate(element=>getComputedStyle(element).scrollBehavior)).toBe('auto');
    await page.emulateMedia({media:'print'}); await page.evaluate(()=>window.dispatchEvent(new Event('beforeprint'))); await expect(page.locator('.resource-detail details').first()).toHaveAttribute('open','');
  });
  test('authentication handoff, isolated failures, mobile containment and axe', async ({page})=>{
    await page.setViewportSize({width:375,height:812}); const state=classification=>({classification,server_observed_at:envelope.server_observed_at,evidence:{...envelope,status:classification},reason:classification});
    await connectedPage(page,{'/operator-ui/api/v1/overview':{classification:'NON_OPERATIONAL/AUTHENTICATION_REQUIRED'},'/operator-ui/api/v1/races/upcoming':state('UNAVAILABLE/DATA_MISSING'),'/operator-ui/api/v1/predictions/recent':state('INVALID/INTEGRITY_FAILED'),'/operator-ui/api/v1/collector':state('DIVERGENT'),'/operator-ui/api/v1/corpus':'OFFLINE','/operator-ui/api/v1/models':state('STALE')});
    await expect(page.getByRole('link',{name:'Open secure login'})).toHaveAttribute('href','/operator-ui/login');
    for(const value of ['UNAVAILABLE/DATA_MISSING','INVALID/INTEGRITY_FAILED','DIVERGENT','NON_OPERATIONAL/OFFLINE','STALE']) await expect(page.getByText(value).first()).toBeVisible();
    await expect(page.locator('[data-resource="corpus"] details summary')).toContainText('request observed not supplied');
    await expectNoOverflow(page); expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);
  });
  test('exact-race readiness completes before background resource fanout', async ({page})=>{
    const available=data=>({classification:'AVAILABLE/FRESH',server_observed_at:envelope.server_observed_at,evidence:envelope,data});
    const unavailable={classification:'UNAVAILABLE/DATA_MISSING',server_observed_at:envelope.server_observed_at,evidence:{...envelope,status:'UNAVAILABLE/DATA_MISSING'},reason:'UNAVAILABLE/DATA_MISSING'};
    let backgroundStarted=false;
    const background=async()=>{backgroundStarted=true;await new Promise(resolve=>setTimeout(resolve,75));return available({});};
    const race=async()=>{await new Promise(resolve=>setTimeout(resolve,25));return backgroundStarted?unavailable:available({races:[{route_id:'race-route-1',race_id:'exact-race-1',venue:'EXACT',race_number:1,jump_utc:'2099-04-01T10:30:00Z',runner_set_sha256:'c'.repeat(64),runners:[{box:1,name:'Exact Runner'}]}]});};
    await connectedPage(page,{
      '/operator-ui/api/v1/races/upcoming':race,
      '/operator-ui/api/v1/models':available({models:[]}),
      '/operator-ui/api/v1/overview':background,
      '/operator-ui/api/v1/predictions/recent':background,
      '/operator-ui/api/v1/collector':background,
      '/operator-ui/api/v1/corpus':background,
      '/operator-ui/api/v1/system':background,
      '/operator-ui/api/v1/audit':background,
    });
    await expect(page.locator('[data-resource="upcoming-races"] .resource-state')).toHaveText('AVAILABLE/FRESH');
    await expect(page.getByText('exact-race-1').first()).toBeVisible();
  });
});
