(() => {
  'use strict';
  const node=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);return e;};
  const status=document.querySelector('#forecast-status'), sources=document.querySelector('#forecast-sources'), programme=document.querySelector('#programme-status'), collector=document.querySelector('#collector-status');
  const time=value=>value?new Date(value).toLocaleString('en-AU',{timeZone:'Australia/Melbourne',timeZoneName:'short',hour12:false}):'Not available';
  const percent=value=>{if(typeof value!=='number'||!Number.isFinite(value))throw new Error('Missing probability');return `${(100*value).toFixed(4)}%`;};
  function forecast(value){
    const race=value.race, article=node('article');article.className='panel retained-forecast';article.dataset.predictionId=value.prediction_id;article.dataset.jump=value.race.jump_timestamp;
    const venues={GEE:'Geelong',HOR:'Horsham',BULLI:'Bulli'};
    article.append(node('h3',`${venues[race.venue]||race.venue} R${race.race_number} · ${race.race_date}`),node('p',`${value.verification} · ${value.temporal_status==='HISTORICAL'?'Historical forecast — scheduled jump has passed':'Pre-jump forecast'} · Jump ${time(race.jump_timestamp)}`));
    if(value.verification!=='VERIFIED'||!Array.isArray(value.runners)||!value.runners.length)throw new Error('Unverified forecast');
    value.runners.forEach(r=>{percent(r.model_probability);percent(r.market_probability);if(r.model_probability<0||r.model_probability>1||r.market_probability<0||r.market_probability>1)throw new Error('Invalid probability');});
    const top=[...value.runners].sort((a,b)=>a.model_rank-b.model_rank)[0];
    const summary=node('p',`Highest model probability: ${top.name} (${(100*top.model_probability).toFixed(1)}%). Captured market: ${(100*top.market_probability).toFixed(1)}%.`);summary.className='forecast-leader';article.append(summary);
    article.append(node('p','Difference is model probability minus market probability, shown in percentage points (pp). Probabilities are estimates, not guaranteed outcomes.'));

    const table=node('table'),head=node('thead'),tr=node('tr');
    ['Box','Runner','Captured WIN odds','Normalized market probability','Model probability','Model rank','Difference (pp)'].forEach(v=>tr.append(node('th',v)));head.append(tr);table.append(head);const body=node('tbody');
    value.runners.forEach(r=>{if(typeof r.win_odds!=='number'||!Number.isFinite(r.win_odds))throw new Error('Missing odds');const row=node('tr');[r.box,r.name,r.win_odds.toFixed(2),percent(r.market_probability),percent(r.model_probability),r.model_rank,`${(100*(r.model_probability-r.market_probability)).toFixed(2)} pp`].forEach(v=>row.append(node('td',v)));
      [r.market_probability,r.model_probability].forEach((probability,index)=>{const meter=node('span');meter.className=`probability-meter ${index?'probability-meter--model':'probability-meter--market'}`;meter.style.setProperty('--probability',`${100*probability}%`);meter.setAttribute('aria-hidden','true');row.children[index+3].append(meter);});body.append(row);});table.append(body);const scroll=node('div');scroll.className='table-scroll';scroll.append(table);article.append(scroll);
    const details=node('details');details.append(node('summary','Timestamps and forecast evidence'),node('p',`Quote: ${time(value.quote_at)} · Prediction: ${time(value.prediction_at)} · Verified: ${time(value.verification_at)}`),node('p',`Independent verification: ${time(value.independent_verification_at)} · Independent chain audit: ${time(value.independent_chain_audit_at)}`),node('p',`Forecast ${value.prediction_id}`),node('p',`Model ${value.model.identity}`),node('p',`Model SHA-256 ${value.model.sha256}`),node('p',`Bundle manifest SHA-256 ${value.manifest_sha256}`));article.append(details);return article;
  }
  let loading=false;
  async function load(){
    if(loading)return;loading=true;
    sources.replaceChildren();programme.replaceChildren();collector.replaceChildren();status.textContent='Checking retained artifacts…';
    try{
      const response=await fetch('/operator-ui/api/v1/predictions/retained',{credentials:'same-origin',cache:'no-store'});
      if(response.status===401){status.textContent='Authentication required. ';const login=node('a','Sign in to view forecasts');login.href='/operator-ui/sign-in';status.append(login);return;}
      if(!response.ok)throw new Error('Forecast API unavailable');const data=await response.json();
      if(data.schema!=='operator_ui_retained_forecasts_v1'||!Array.isArray(data.sources))throw new Error('Invalid forecast response');
      status.textContent=`Artifacts checked ${time(data.observed_at)}. Times shown in Melbourne time.`;
      programme.append(node('p',data.programme.state.replaceAll('_',' ')),node('p',`Next session: ${time(data.programme.next_session)} – ${time(data.programme.next_session_end)}`),node('p',`Collection active: ${data.programme.collecting===true?'Yes':data.programme.collecting===false?'No':'Unknown'}`));
      if(data.programme.reason)programme.append(node('p',data.programme.reason));
      if(data.programme.health_status)programme.append(node('p',`Last programme status: ${data.programme.health_status} · ${time(data.programme.observed_at)}`));
      const rendered=node('div');
      for(const source of data.sources){const section=node('section');section.append(node('h2',source.source==='operational'?'Operational forecasts':source.source==='engineering'?'Engineering rehearsals':'October programme forecasts'),node('p',source.state));if(source.reason)section.append(node('p',source.reason));for(const value of source.forecasts){try{section.append(forecast(value));}catch(_){section.append(node('p','A forecast was withheld because its runner probabilities could not be validated.'));}}for(const error of source.errors)section.append(node('p',`${error.prediction_id}: ${error.reason}`));rendered.append(section);}
      sources.replaceChildren(...rendered.children);
      const fresh=await fetch('/operator-ui/api/v1/collector',{credentials:'same-origin',cache:'no-store'});
      if(!fresh.ok)throw new Error('Collector evidence unavailable');const value=await fresh.json();
      collector.append(node('p',`Current collector evidence: ${value.classification}`),node('p','This readiness status does not change the validity of verified historical forecasts.'));
      if(value.reason)collector.append(node('p',value.reason));
    }catch(error){if(!sources.childElementCount){sources.replaceChildren();status.textContent='Forecasts unavailable. Retained evidence could not be loaded or validated.';}collector.textContent='Current collector readiness unavailable. No freshness claim is made.';}finally{loading=false;}
  }
  setInterval(()=>{if(!document.hidden)load();},60000);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)load();});
  document.querySelector('a[href="#forecast-system"]').addEventListener('click',()=>{document.querySelector('#forecast-system').open=true;});
  document.querySelector('#refresh-forecasts').addEventListener('click',load);load();
})();
