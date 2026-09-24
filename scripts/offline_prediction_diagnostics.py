"""Post-fit uncertainty and descriptive diagnostics; never fits/tunes a model."""
import json
from pathlib import Path
import numpy as np
from scripts.offline_prediction_research import group, scores, paired, write, SEED

def diagnose(out):
 rows=[json.loads(l) for l in (out/'predictions.jsonl').open()];start,end=group(rows);days=np.array([rows[i]['race_date'] for i in start]);dates=sorted(set(days));market=np.array([r['market'] for r in rows]);rng=np.random.default_rng(SEED)
 bymodel={}
 for name in rows[0]['predictions']:
  p=np.array([r['predictions'][name] for r in rows]);delta=scores(rows,p)[0]-scores(rows,market)[0];date_blocks=[delta[days==d] for d in dates]
  # Three consecutive observed-date blocks, circular resampling; sensitivity only.
  sims=[]
  for _ in range(2000):
   selected=[]
   for anchor in rng.integers(0,len(dates),int(np.ceil(len(dates)/3))):selected.extend([(anchor+j)%len(dates) for j in range(3)])
   sims.append(float(np.concatenate([date_blocks[j] for j in selected[:len(dates)]]).mean()))
  bymodel[name]={'date_deltas':{d:{'races':int(sum(days==d)),'mean_delta':float(delta[days==d].mean()),'sum_delta':float(delta[days==d].sum())} for d in dates},'negative_date_blocks':int(sum(b.mean()<0 for b in date_blocks)),'date_blocks':len(dates),'leave_one_date_out_delta_range':[float(min(delta[days!=d].mean() for d in dates)),float(max(delta[days!=d].mean() for d in dates))],'moving_3_observed_dates_ci95':np.quantile(sims,[.025,.975]).tolist()}
 comparisons={}
 for a,b in [('residual_with_box','residual_frozen_method'),('residual_ewma','residual_frozen_method'),('residual_without_context','residual_frozen_method'),('residual_without_recent','residual_frozen_method'),('residual_half','residual_frozen_method')]:
  comparisons[a+' minus '+b]=paired(rows,np.array([r['predictions'][a] for r in rows]),np.array([r['predictions'][b] for r in rows]))
 # Calibration uncertainty: bootstrap all dates, including dates without selection.
 selections={};favourites=[]
 for a,b in zip(start,end):
  rr=rows[a:b];m=max(r['market'] for r in rr);ff=[r for r in rr if r['market']==m]
  if len(ff)==1:favourites.extend(ff)
 rules={'unique_favourites':favourites,'favourite_primary_minus_005':[r for r in favourites if r['predictions']['residual_frozen_method']-r['market']<=-.05], 'outsider_primary_odds10_plus002':[r for r in rows if r['odds']>=10 and r['predictions']['residual_frozen_method']-r['market']>=.02], 'all_outsiders_odds10':[r for r in rows if r['odds']>=10], 'outsider_sensitivity_odds10_plus0005':[r for r in rows if r['odds']>=10 and r['predictions']['residual_frozen_method']-r['market']>=.005]}
 for name,rr in rules.items():
  sims=[];missing=0;blocks=[[r for r in rr if r['race_date']==d] for d in dates]
  for _ in range(2000):
   sample=[r for i in rng.integers(0,len(dates),len(dates)) for r in blocks[i]]
   if not sample:missing+=1;continue
   sims.append(np.mean([r['y']-r['market'] for r in sample]))
  selections[name]={'n':len(rr),'wins':sum(r['y'] for r in rr),'dates_with_selections':len({r['race_date'] for r in rr}),'mean_actual_minus_market':float(np.mean([r['y']-r['market'] for r in rr])) if rr else None,'date_bootstrap_ci95':np.quantile(sims,[.025,.975]).tolist() if sims and len({r['race_date'] for r in rr})>=2 else None,'zero_selection_replicates':missing,'warning':'small selected samples yield unstable bootstrap intervals; descriptive, unadjusted'}
 write(out/'postfit_diagnostics.json',{'models':bymodel,'paired_feature_comparisons':comparisons,'selection_calibration':selections,'status':'POST_FIT_DIAGNOSTIC_NO_SELECTION_OR_REFIT','source_predictions_sha256':__import__('hashlib').sha256((out/'predictions.jsonl').read_bytes()).hexdigest()})

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);diagnose(p.parse_args().out)
