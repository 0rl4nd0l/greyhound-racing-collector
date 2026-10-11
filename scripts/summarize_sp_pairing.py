#!/usr/bin/env python3
"""Fixed paired date-cluster summaries; descriptive, never a fresh holdout claim."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(path):
    scores=json.loads((path/'race_scores.json').read_text())
    later={}
    for row in scores:
        if row['partition']=='later':later.setdefault(row['race'],{})[row['model']]=row
    names=sorted(set.intersection(*[set(x) for x in later.values()]))
    dates=sorted({next(iter(x.values()))['date'] for x in later.values()})
    rng=np.random.default_rng(20261011);draws=rng.integers(0,len(dates),size=(2000,len(dates)))
    result={}
    for name in names:
        if name=='market_calibrated':continue
        delta=[]
        for key,rs in later.items():
            delta.append({'race':key,'date':rs[name]['date'],'track':rs[name]['track'],'delta':rs[name]['log_loss']-rs['market_calibrated']['log_loss']})
        sums=np.array([sum(x['delta'] for x in delta if x['date']==date) for date in dates]);counts=np.array([sum(x['date']==date for x in delta) for date in dates])
        boots=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        loo=(sums.sum()-sums)/(counts.sum()-counts)
        result[name]={'reference':'market_calibrated','races':len(delta),'dates':len(dates),'log_loss_delta':float(sums.sum()/counts.sum()),'date_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),'leave_one_date_out_range':[float(loo.min()),float(loo.max())],'improved_races':sum(x['delta']<-1e-8 for x in delta),'worsened_races':sum(x['delta']>1e-8 for x in delta),'unchanged_races':sum(abs(x['delta'])<=1e-8 for x in delta),'by_date':{d:{'races':int(n),'delta':float(s/n),'omitting_date_delta':float(l)} for d,s,n,l in zip(dates,sums,counts,loo)}}
    (path/'paired_summary.json').write_text(json.dumps({'seed':20261011,'replicates':2000,'interpretation':'Descriptive only: exposed10dates, repeated dogs and model selection not modeled','contrasts':result},indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:{i:v for i,v in x.items() if i!='by_date'} for k,x in result.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);summarize(p.parse_args().directory)
