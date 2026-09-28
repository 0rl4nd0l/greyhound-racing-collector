#!/usr/bin/env python3
"""Isolated offline development experiment. No network, production imports or writes."""
from __future__ import annotations
import argparse, collections, csv, hashlib, importlib.metadata, json, math, os, platform, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/l4nd0/greyhound/artifacts/sportsbet_win_market_surface_audit_20260815_report_only/canonical_win_matrix.jsonl')
CLOSED=Path('/home/l4nd0/greyhound-retrospective-diagnostic-20260916/pre-outcome-manifest-v2.json')
AUGUST=Path('/home/l4nd0/greyhound_racing_collector/artifacts/august_untouched_cohort_20260813T181500Z_report_only/frozen_august_odds_cohort.jsonl')
FORM=Path('/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-form-only-v1-acquisition-20260718/reports/agent_jobs/form_only_v1_acquisition_foundation_20260718')
FROZEN_FEATURES=('prior_start_count','days_since_last_start','recent_finish_mean_3','recent_finish_best_5','recent_win_rate_5','recent_place_rate_5','recent_avg_margin_5','career_win_rate','career_place_rate','career_avg_finish','starts_same_venue','win_rate_same_venue','starts_same_distance','win_rate_same_distance','same_grade_start_count','same_grade_win_rate')
FOLDS=[dict(name='period1',train_end='2026-06-17',val_start='2026-06-18',val_end='2026-06-21',test_start='2026-06-24',test_end='2026-06-30'),dict(name='period2',train_end='2026-06-24',val_start='2026-06-25',val_end='2026-06-30',test_start='2026-07-01',test_end='2026-07-02'),dict(name='final_later_block',train_end='2026-06-30',val_start='2026-07-01',val_end='2026-07-02',test_start='2026-07-03',test_end='2026-07-09')]
SEED=20260924

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def scalar(line,key):
 m=re.search(r'"'+re.escape(key)+r'"\s*:\s*("(?:[^"\\]|\\.)*"|null|[-0-9.]+)',line)
 if not m:raise ValueError('missing identity '+key)
 return json.loads(m[1])
def token(x):return re.sub('[^A-Z0-9]','',x.upper())
def race_key(rid):
 m=re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})',rid)
 if not m:raise ValueError('unsupported race identity')
 return f'{m[3]}|{m[2]}|{int(m[1])}'
def protected():
 closed=json.loads(CLOSED.read_text()); keys={r['race_key']:'closed_114' for r in closed['candidates']}
 with (FORM/'out_of_time_races.csv').open() as f:
  for r in csv.DictReader(f):keys[race_key(r['race_id'])]='reserved_form_only_out_of_time'
 # Project only race identifiers; outcome columns are never decoded.
 for line in AUGUST.open():
  rid=scalar(line,'race_id');keys[race_key(rid)]='reserved_august_frozen_odds'
 return keys

def prepare(out):
 from offline_form_packet import load_features
 start=time.monotonic();out.mkdir(parents=True,exist_ok=False)
 deny=protected()
 if min(key[:10] for key in deny)<='2026-07-09':raise ValueError('protected history could precede target: new scoped history exclusion required')
 write(out/'protected_records.json',{'records':deny,'windows':[{'start':'2026-08-18','end':'2026-09-30','authority':'docs/sportsbet_betfair_forward_consensus_protocol.md; predecessor also excluded'},{'start':'2026-10-01','end':None,'authority':'docs/forward_overround_successor_protocol.md'}],'additional_scope_cap':'all targets after 2026-07-09 and histories on/after target date excluded'})
 assert sha(SOURCE)=='eb1783d4cc07e6980463a097c97fdac9f5370b08f493ca15addf768aa0b014b6', 'canonical source drift'
 forms,provenance=load_features(excluded_race_keys=deny);write(out/'form_provenance.json',provenance)
 sidecar_path=SOURCE.with_name('canonical_win_sidecar.jsonl')
 assert sha(sidecar_path)=='880ae93680e56991fa2c9eb316732cbc71bc7ff713525efcf83750ceace4493d'
 sidecars={}
 for line in sidecar_path.open():
  rid=scalar(line,'race_id');day=scalar(line,'race_date')
  if race_key(rid) not in deny and '2026-06-10'<=day<='2026-07-09':
   v=json.loads(line);key=(rid,int(v['box_number']),v['capture_timestamp'])
   assert key not in sidecars,'duplicate market provenance'
   sidecars[key]=v
 fm={(r['race_id'],int(r['box_number'])):r for r in forms};assert len(fm)==len(forms)
 groups=collections.defaultdict(list);exclusions=[];source_dates=collections.Counter()
 for line in SOURCE.open():
  rid=scalar(line,'race_id');day=scalar(line,'race_date');source_dates[day]+=1
  if race_key(rid) in deny or not '2026-06-10'<=day<='2026-07-09':
   exclusions.append({'race_id':rid,'reason':deny.get(race_key(rid),'outside_declared_development_interval')});continue
  r=json.loads(line);v=sidecars[(rid,int(r['box_number']),r['odds_capture_timestamp'])]
  assert v['source']=='sportsbet' and v['race_qualified'] is True
  assert v['canonical_win_odds']==r['canonical_sportsbet_win_odds']==v['paired_win_odds']
  assert v['raw_runner_text_sha256']==r['sportsbet_win_raw_sha256']
  assert v['sidecar_row_sha256']==r['sportsbet_win_sidecar_row_sha256']
  groups[rid].append(r)
 rows=[];included=[]
 for rid,rr in sorted(groups.items()):
  reasons=[];boxes=[int(r['box_number']) for r in rr]
  if len(boxes)!=len(set(boxes)):reasons.append('duplicate_box')
  if any(int(r['field_size'])!=len(rr) for r in rr):reasons.append('incomplete_field')
  if sum(r['label_is_winner'] for r in rr)!=1 or any(r['label_is_winner'] not in (0,1) for r in rr):reasons.append('not_single_winner')
  if any(not 1<=int(r['label_finish_position'])<=len(rr) for r in rr):reasons.append('scratch_or_invalid_finish')
  lags=[(datetime.fromisoformat(r['jump_at'])-datetime.fromisoformat(r['odds_capture_timestamp'])).total_seconds()/60 for r in rr]
  if any(not 2<=v<=10 for v in lags):reasons.append('no_snapshot_in_T2_to_T10_window')
  if len({r['odds_capture_timestamp'] for r in rr})!=1:reasons.append('non_atomic_snapshot')
  ff=[fm.get((rid,b)) for b in boxes]
  if any(f is None for f in ff):reasons.append('incomplete_verified_pre_race_form')
  if not reasons:
   if any(token(r['dog_name'])!=f['dog_token'] for r,f in zip(rr,ff)):reasons.append('runner_identity_mismatch')
   if any(int(f['features']['field_size'])!=len(rr) for f in ff):reasons.append('form_field_changed')
   if any(datetime.fromisoformat(f['jump_timestamp'])!=datetime.fromisoformat(r['jump_at']) for f,r in zip(ff,rr)):reasons.append('scheduled_jump_disagreement')
   if any(race_key(rid) in deny for _ in [0]):raise AssertionError('protected')
  if reasons:exclusions.append({'race_id':rid,'reason':';'.join(reasons)});continue
  odds=np.array([r['canonical_sportsbet_win_odds'] for r in rr],float)
  if np.any(~np.isfinite(odds)) or np.any(odds<=1):raise ValueError('invalid odds')
  prob=(1/odds)/(1/odds).sum()
  for r,f,p,lag in zip(rr,ff,prob,lags):
   features=f['features']
   rows.append(dict(race_id=rid,race_date=r['race_date'],venue=r['venue'],box=int(r['box_number']),dog_token=f['dog_token'],y=r['label_is_winner'],odds=r['canonical_sportsbet_win_odds'],market=float(p),capture=r['odds_capture_timestamp'],jump=r['jump_at'],capture_lead_minutes=lag,features=features))
  included.append(rid)
 rows.sort(key=lambda r:(r['race_date'],r['race_id'],r['box']))
 with (out/'development.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
 # Deduplicate identity-only exclusions, including protected rows excluded before decode.
 exclusions=list({(e['race_id'],e['reason']):e for e in exclusions}.values());write(out/'exclusions.json',exclusions)
 fs=sorted(set().union(*(r['features'] for r in rows)))
 assessment={'races':len(included),'runners':len(rows),'date_min':min(r['race_date'] for r in rows),'date_max':max(r['race_date'] for r in rows),'dates':dict(collections.Counter(r['race_date'] for r in rows)),'venue_races':dict(collections.Counter(next(r['venue'] for r in rows if r['race_id']==rid) for rid in included)),'source_identity_only_date_runner_counts':dict(source_dates),'exclusion_races_by_reason':dict(collections.Counter(e['reason'] for e in exclusions)),'missingness':{f:sum(r['features'].get(f) is None for r in rows)/len(rows) for f in fs},'capture_lead_quantiles_minutes':np.quantile([r['capture_lead_minutes'] for r in rows],[0,.25,.5,.75,1]).tolist(),'preparation_seconds':time.monotonic()-start,'inputs':{str(p):sha(p) for p in [SOURCE,sidecar_path,CLOSED,AUGUST,FORM/'out_of_time_races.csv']},'development_sha256':sha(out/'development.jsonl')}
 write(out/'dataset_assessment.json',assessment)
 protocol={'status':'FROZEN_BEFORE_COMPARATIVE_PERFORMANCE','folds':FOLDS,'seed':SEED,'cutoff':'T-2 minutes, retained latest canonical snapshot age at decision <=8min; no later prices substituted','population':'complete corrected WIN and verified raw-card intersection, June10-July09, exact protected exclusions before outcome decode','features':'raw card histories strictly before target; pre-race card captured >=60min before jump; preprocessing fit train only','models':['uniform','market','market_power_calibrated','form_regularized','residual_frozen_method','residual_half','residual_without_recent','residual_without_context','residual_recent_only','residual_ewma','residual_with_box','form_boosted','market_plus_boosted'],'search_budget':'one fixed tree (60 iterations, 7 leaves, min leaf30,l2=10); offset objective mean race loss +0.5||beta||^2; form-only mean race loss +0.5||beta||^2/N_train_races; validation-only market exponent [0.5,1.75], form temperature [0.5,2], tree blend in {0,.1,.25,.5}; no population or feature tuning','selection_rules':'unique favourite model-minus-market <= -.005/-.01/-.02/-.05/-.10; outsider decimal WIN>=8/10/15 and model-minus-market>=.005/.01/.02/.05. Primary -.05 and >=10 with +.02. Thresholds specified before metrics, not chosen by ROI. All later races count including no selection.','uncertainty':'2000 paired calendar-date block bootstrap, seed20260924; descriptive intervals unadjusted for multiple comparisons','returns':'not calculated: deductions, scratch timing, accepted execution and price availability at T-2 unverified','holdout':'July03-09 held back within this run; historically inspected, exploratory not pristine'}
 write(out/'protocol.json',protocol);print(json.dumps(assessment,indent=2))

def group(rows):
 ids=np.array([r['race_id'] for r in rows]); starts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1];return starts,np.r_[starts[1:],len(rows)]
def probabilities(rows,logits):
 starts,ends=group(rows);out=np.empty(len(rows))
 for a,b in zip(starts,ends):out[a:b]=np.exp(logits[a:b]-logsumexp(logits[a:b]))
 return out

def prefit(rows,features):
 x=np.array([[r['features'].get(f) if r['features'].get(f) is not None else np.nan for f in features] for r in rows],float)
 med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x.T]);a=np.c_[np.where(np.isfinite(x),x,med),~np.isfinite(x)];mean=a.mean(0);std=a.std(0);std[std<1e-12]=1
 return dict(features=features,med=med,mean=mean,std=std)
def transform(rows,prep,center=True):
 x=np.array([[r['features'].get(f) if r['features'].get(f) is not None else np.nan for f in prep['features']] for r in rows],float);a=np.c_[np.where(np.isfinite(x),x,prep['med']),~np.isfinite(x)];a=(a-prep['mean'])/prep['std']
 if center:
  for s,e in zip(*group(rows)):a[s:e]-=a[s:e].mean(0)
 return a

def fit_linear(rows,features,offset=True,cap=.35):
 prep=prefit(rows,features);x=transform(rows,prep);y=np.array([r['y'] for r in rows]);base=np.log([r['market'] for r in rows]) if offset else np.zeros(len(rows));n=len(group(rows)[0])
 def obj(beta):
  z=x@beta;res=cap*np.tanh(z/cap) if cap else z;deriv=1-np.tanh(z/cap)**2 if cap else np.ones(len(z));p=probabilities(rows,base+res)
  penalty=1.0 if offset else 1.0/n
  loss=-np.log(p[y==1]).mean()+.5*penalty*np.dot(beta,beta)
  grad=x.T@((p-y)*deriv)/n+penalty*beta
  return loss,grad
 fit=minimize(obj,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12,'gtol':1e-8,'maxls':50})
 if not fit.success:raise RuntimeError(str(fit.message))
 return dict(prep=prep,beta=fit.x,offset=offset,cap=cap,iterations=fit.nit)
def predict_linear(rows,model,strength=1):
 z=transform(rows,model['prep'])@model['beta'];cap=model['cap'];res=cap*np.tanh(z/cap) if cap else z
 base=np.log([r['market'] for r in rows]) if model['offset'] else np.zeros(len(rows))
 return probabilities(rows,base+strength*res)
def scores(rows,p):
 y=np.array([r['y'] for r in rows]);start,end=group(rows);ll=[];bs=[];top=[];rank=[]
 for a,b in zip(start,end):
  z=p[a:b];yy=y[a:b];winner=int(np.argmax(yy));ll.append(float(-np.log(z[winner])));bs.append(float(((z-yy)**2).sum()));top.append(float(yy[z==z.max()].mean()));rank.append(float(1+sum(z>z[winner])+.5*(sum(z==z[winner])-1)))
 return np.array(ll),np.array(bs),np.array(top),np.array(rank)
def metrics(rows,p):
 ll,bs,top,rank=scores(rows,p);y=np.array([r['y'] for r in rows]);bins=[]
 for low in np.arange(0,1,.1):
  m=(p>=low)&(p<low+.1)
  if m.any():bins.append(dict(low=float(low),n=int(m.sum()),predicted=float(p[m].mean()),observed=float(y[m].mean())))
 return dict(races=len(ll),runners=len(rows),log_loss=float(ll.mean()),brier=float(bs.mean()),top1=float(top.mean()),winner_rank=float(rank.mean()),runner_ece=float(sum(b['n']*abs(b['predicted']-b['observed']) for b in bins)/len(rows)),calibration=bins)
def paired(rows,p,ref):
 a,b,_,_=scores(rows,p);c,d,_,_=scores(rows,ref);dates=[rows[i]['race_date'] for i in group(rows)[0]];days=sorted(set(dates));blocks=[np.flatnonzero(np.array(dates)==day) for day in days];rng=np.random.default_rng(SEED);deltas=[]
 for _ in range(2000):
  idx=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]);deltas.append([(a-c)[idx].mean(),(b-d)[idx].mean()])
 ci=np.quantile(deltas,[.025,.975],axis=0)
 return dict(delta_log_loss=float((a-c).mean()),ll_ci95=ci[:,0].tolist(),delta_brier=float((b-d).mean()),brier_ci95=ci[:,1].tolist(),date_blocks=len(days))

def evaluate(out):
 start=time.monotonic();protocol=json.loads((out/'protocol.json').read_text());assert protocol['folds']==FOLDS,'frozen split drift';assert not (out/'results.json').exists(),'outputs exist'
 data=[json.loads(l) for l in (out/'development.jsonl').open()];assert sha(out/'development.jsonl')==json.loads((out/'dataset_assessment.json').read_text())['development_sha256']
 predictions=[];fits=[];models=['uniform','market','market_power_calibrated','form_regularized','residual_frozen_method','residual_half','residual_without_recent','residual_without_context','residual_recent_only','residual_ewma','residual_with_box','form_boosted','market_plus_boosted']
 for fold in FOLDS:
  train=[r for r in data if r['race_date']<=fold['train_end']];val=[r for r in data if fold['val_start']<=r['race_date']<=fold['val_end']];test=[r for r in data if fold['test_start']<=r['race_date']<=fold['test_end']]
  if min(len(train),len(val),len(test))==0:raise ValueError('empty fold')
  tv={r['race_id'] for r in train};vv={r['race_id'] for r in val};tt={r['race_id'] for r in test};assert not (tv&vv or tv&tt or vv&tt)
  pm=np.array([r['market'] for r in test]);pv=np.array([r['market'] for r in val]);result={'market':pm,'uniform':probabilities(test,np.zeros(len(test)))}
  power=minimize_scalar(lambda a:scores(val,probabilities(val,a*np.log(pv)))[0].mean(),bounds=(.5,1.75),method='bounded').x
  result['market_power_calibrated']=probabilities(test,power*np.log(pm))
  frozen=fit_linear(train,list(FROZEN_FEATURES));result['residual_frozen_method']=predict_linear(test,frozen);result['residual_half']=predict_linear(test,frozen,.5)
  # Fixed feature ablations, not feature selection on evaluation data.
  variants={'residual_without_recent':[f for f in FROZEN_FEATURES if not f.startswith('recent_')],'residual_without_context':[f for f in FROZEN_FEATURES if not ('same_' in f)],'residual_recent_only':[f for f in FROZEN_FEATURES if f.startswith('recent_') or f in ['prior_start_count','days_since_last_start']]}
  variants['residual_ewma']=[f for f in FROZEN_FEATURES if not f.startswith('recent_')]+['ewma_finish_half_life_3_starts','ewma_margin_half_life_3_starts']
  variants['residual_with_box']=list(FROZEN_FEATURES)+['box_number']
  saved={'residual_frozen_method':frozen}
  for name,features in variants.items():
   m=fit_linear(train,features);saved[name]=m;result[name]=predict_linear(test,m)
  form=fit_linear(train,list(FROZEN_FEATURES),False,None);saved['form_regularized']=form
  vform=predict_linear(val,form);temp=minimize_scalar(lambda a:scores(val,probabilities(val,a*np.log(vform)))[0].mean(),bounds=(.5,2),method='bounded').x
  result['form_regularized']=probabilities(test,temp*np.log(predict_linear(test,form)))
  prep=prefit(train,list(FROZEN_FEATURES));tree=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=SEED)
  tree.fit(transform(train,prep),[r['y'] for r in train]);pt=probabilities(test,np.log(np.clip(tree.predict_proba(transform(test,prep))[:,1],1e-8,1)));vt=probabilities(val,np.log(np.clip(tree.predict_proba(transform(val,prep))[:,1],1e-8,1)))
  result['form_boosted']=pt
  blend_scores={str(w):float(scores(val,(1-w)*pv+w*vt)[0].mean()) for w in [0,.1,.25,.5]};weight=float(min(blend_scores,key=blend_scores.get));result['market_plus_boosted']=(1-weight)*pm+weight*pt
  fits.append(dict(fold=fold,train_races=len(tv),validation_races=len(vv),test_races=len(tt),train_runners=len(train),validation_runners=len(val),test_runners=len(test),market_power=float(power),form_temperature=float(temp),tree_blend_weight=weight,tree_blend_validation_losses=blend_scores,linear_models={k:json.loads(json.dumps(m,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x)) for k,m in saved.items()}))
  for i,r in enumerate(test):predictions.append({**r,'period':fold['name'],'predictions':{name:float(p[i]) for name,p in result.items()}})
  print('completed',fold['name'],len(tt),'races',flush=True)
 pmarket=np.array([r['market'] for r in predictions]);overall={};periods={}
 for name in models:
  p=np.array([r['predictions'][name] for r in predictions]);overall[name]={**metrics(predictions,p),**paired(predictions,p,pmarket)}
  periods[name]={}
  for fold in FOLDS:
   rr=[r for r in predictions if r['period']==fold['name']];pp=np.array([r['predictions'][name] for r in rr]);ref=np.array([r['market'] for r in rr]);periods[name][fold['name']]={**metrics(rr,pp),**paired(rr,pp,ref)}
 with (out/'predictions.jsonl').open('w') as f:
  for r in predictions:f.write(json.dumps(r,sort_keys=True)+'\n')
 write(out/'fits.json',fits);write(out/'results.json',dict(overall=overall,periods=periods,seconds=time.monotonic()-start,protocol_sha256=sha(out/'protocol.json'),environment={'python':sys.executable,'version':sys.version,'platform':platform.platform(),'packages':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn']},'thread_env':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'runner_sha256':sha(Path(__file__))}))
 investigate(out,predictions)
 print(json.dumps({k:{f:v[f] for f in ['log_loss','brier','top1','delta_log_loss','ll_ci95']} for k,v in overall.items()},indent=2))

def investigate(out,rows):
 start,end=group(rows);fav=[];winners=[];selections=[]
 for a,b in zip(start,end):
  rr=rows[a:b];market=np.array([r['market'] for r in rr]);idx=np.flatnonzero(market==market.max());winner=next(r for r in rr if r['y']);winners.append(winner['odds'])
  if len(idx)==1:fav.append(rr[int(idx[0])])
 def stats(selected,total):
  if not selected:return {'selections':0,'eligible_races':total,'selected_races':0,'no_selection_races':total}
  n=len(selected);nr=len({r['race_id'] for r in selected});return dict(selections=n,eligible_races=total,selected_races=nr,no_selection_races=total-nr,selection_race_fraction=nr/total,wins=sum(r['y'] for r in selected),actual_win_rate=sum(r['y'] for r in selected)/n,mean_market_probability=float(np.mean([r['market'] for r in selected])),mean_model_probability=float(np.mean([r['predictions']['residual_frozen_method'] for r in selected])),odds_quantiles=np.quantile([r['odds'] for r in selected],[0,.25,.5,.75,1]).tolist(),by_period={p:{'selections':sum(r['period']==p for r in selected),'wins':sum(r['y'] for r in selected if r['period']==p),'eligible_races':len({r['race_id'] for r in rows if r['period']==p})} for p in [f['name'] for f in FOLDS]})
 total=len(start)
 for threshold in [.005,.01,.02,.05,.10]:
  selected=[r for r in fav if r['predictions']['residual_frozen_method']-r['market']<=-threshold];selections.append(dict(rule='vulnerable_favourite',threshold=threshold,**stats(selected,total)))
 for odds in [8,10,15]:
  for edge in [.005,.01,.02,.05]:
   selected=[r for r in rows if r['odds']>=odds and r['predictions']['residual_frozen_method']-r['market']>=edge];selections.append(dict(rule='outsider',minimum_odds=odds,edge=edge,**stats(selected,total)))
 bands=[]
 for lo,hi in [(1,3),(3,5),(5,10),(10,20),(20,10000)]:
  selected=[r for r in rows if lo<=r['odds']<hi];bands.append(dict(low=lo,high=hi,**stats(selected,total)))
 write(out/'favourite_outsider.json',dict(unique_favourites=stats(fav,total),tied_favourite_races=total-len(fav),winner_odds_quantiles=np.quantile(winners,[0,.25,.5,.75,.9,1]).tolist(),winner_odds_ge10=sum(v>=10 for v in winners),rules=selections,all_runner_odds_bands=bands,returns='NOT_COMPUTED_EXECUTION_AND_DEDUCTION_EVIDENCE_MISSING'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','evaluate']);p.add_argument('--out',type=Path,required=True);a=p.parse_args();globals()[a.command](a.out)
