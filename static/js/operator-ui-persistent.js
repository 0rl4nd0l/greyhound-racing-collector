(() => {
  'use strict';
  const status=document.querySelector('#persistent-status'), target=document.querySelector('#persistent-data');
  if(!status||!target)return;
  const node=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);return e;};
  const time=value=>new Date(value).toLocaleString('en-AU',{timeZone:'Australia/Melbourne',timeZoneName:'short',hour12:false});
  const models=['production','market','residual_box','residual_half'];
  const names={production:'Production model',market:'Normalized WIN market',residual_box:'Residual + box',residual_half:'Residual half'};
  const percent=p=>{if(typeof p!=='number'||!Number.isFinite(p)||p<=0||p>=1)throw new Error('Invalid probability');return `${(100*p).toFixed(4)}%`;};
  function table(headers, rows){
    const table=node('table'),head=node('thead'),heading=node('tr'),body=node('tbody');
    headers.forEach(text=>{const cell=node('th',text);cell.scope='col';heading.append(cell);});head.append(heading);table.append(head);
    rows.forEach(values=>{const row=node('tr');values.forEach(value=>row.append(node('td',value)));body.append(row);});table.append(body);
    const scroll=node('div');scroll.className='table-scroll';scroll.append(table);return scroll;
  }
  function forecast(value){
    if(value.evidence_class!=='ENGINEERING'||value.scientific_admission!=='CANARY_NOT_VERIFIED'||!Array.isArray(value.runners)||!value.runners.length)throw new Error('Unverified comparison');
    const article=node('article');article.className='panel persistent-forecast';article.dataset.jobId=value.job_id;
    article.append(node('h3',value.race.race_id),node('p',`Verified four-candidate engineering forecast · Jump ${time(value.race.jump_timestamp)}`));
    const rows=value.runners.map(runner=>[runner.box,runner.name,...models.map(model=>percent(runner.probabilities[model]))]);
    article.append(table(['Box','Runner',...models.map(model=>names[model])],rows));
    const evidence=node('details');evidence.append(node('summary','Forecast evidence'),node('p',`Published ${time(value.published_at)} · Verified ${time(value.verified_at)}`),node('p',`Manifest SHA-256 ${value.manifest_sha256}`));
    models.forEach(model=>evidence.append(node('p',`${names[model]} SHA-256 ${value.models[model]}`)));article.append(evidence);return article;
  }
  let loading=false;
  async function load(){
    if(loading)return;loading=true;target.replaceChildren();status.textContent='Checking persistent collection…';
    try{
      const response=await fetch('/operator-ui/api/v1/predictions/persistent',{credentials:'same-origin',cache:'no-store'});
      if(response.status===401){status.textContent='Sign in to view persistent collection. ';const link=node('a','Sign in');link.href='/operator-ui/sign-in';status.append(link);return;}
      if(!response.ok)throw new Error('Collection unavailable');const data=await response.json();
      if(data.schema!=='operator_ui_persistent_collector_v1')throw new Error('Invalid collection response');
      if(data.state==='UNAVAILABLE'){status.textContent=data.reason;return;}
      status.textContent=`Collector: ${data.state.replaceAll('_',' ')} · Status ${time(data.status_at)} · Inventory ${time(data.inventory_at)} (${data.inventory_state.replaceAll('_',' ').toLowerCase()}).`;
      target.append(node('p',`${data.race_count} races discovered · ${data.upcoming.length} still upcoming. Discovery does not establish fresh runner inputs or forecast readiness.`));
      target.append(node('p','Engineering evidence only. Scientific admission remains CANARY_NOT_VERIFIED.'));
      if(target.dataset.today==='true'){
        const rows=data.upcoming.map(race=>[`${race.venue} R${race.race_number}`,time(race.jump_at),'Waiting for verified capture and forecast']);
        target.append(table(['Race','Scheduled jump (Melbourne)','Forecast readiness'],rows.slice(0,10)));
        if(rows.length>10){const more=node('details');more.append(node('summary',`Show ${rows.length-10} more upcoming races`),table(['Race','Scheduled jump (Melbourne)','Forecast readiness'],rows.slice(10)));target.append(more);}
        const link=node('a','Open verified forecasts');link.href='/operator-ui/forecasts';const paragraph=node('p');paragraph.append(link);target.append(paragraph);
      }else{
        if(!data.forecasts.length)target.append(node('p','No verified forecasts have been published by this collector yet. Historical forecasts remain below.'));
        for(const value of data.forecasts){try{target.append(forecast(value));}catch(_){target.append(node('p','A four-candidate forecast was withheld because its probabilities could not be validated.'));}}
      }
      if(data.forecast_errors.length)target.append(node('p',`${data.forecast_errors.length} completed comparisons could not be verified and were withheld.`));
    }catch(_){status.textContent='Persistent collection unavailable. No freshness claim is made.';target.replaceChildren();}
    finally{loading=false;}
  }
  document.querySelector('#refresh-persistent').addEventListener('click',load);
  setInterval(()=>{if(!document.hidden)load();},60000);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)load();});load();
})();
