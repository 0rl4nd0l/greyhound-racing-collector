"""Fixed reporting of sealed outer predictions; no fitting or selection."""
import collections
import json
from pathlib import Path
import numpy as np
from scripts.offline_prediction_research import write, sha


def race_arrays(rows,p):
    from scripts.offline_systematic_search import starts,validate
    validate(rows,p);s,n=starts(rows);y=np.array([r['y'] for r in rows]);ll=-np.log(p[y==1]);brier=np.add.reduceat((p-y)**2,s)
    top=[];rank=[]
    for a,b in zip(s,n):
        f=p[a:a+b];yy=y[a:a+b];w=int(np.argmax(yy));top.append(float(yy[f==f.max()].mean()));rank.append(float(1+sum(f>f[w])+.5*(sum(f==f[w])-1)))
    return ll,brier,np.array(top),np.array(rank)


def metrics(rows,p):
    ll,bs,top,rank=race_arrays(rows,p);y=np.array([r['y'] for r in rows]);bins=[]
    binid=np.minimum((p*10).astype(int),9)
    for i in range(10):
        mask=binid==i
        if mask.any():bins.append({'lower':i/10,'upper':(i+1)/10,'n':int(mask.sum()),'predicted':float(p[mask].mean()),'actual':float(y[mask].mean())})
    return {'races':len(ll),'runners':len(rows),'log_loss':float(ll.mean()),'brier':float(bs.mean()),'top1':float(top.mean()),'winner_rank':float(rank.mean()),'runner_ece':sum(b['n']*abs(b['predicted']-b['actual']) for b in bins)/len(rows),'calibration_bins':bins}


def selection_stats(rows,flag):
    total=len({r['race_id'] for r in rows});chosen=[r for r in rows if r[flag]];races={r['race_id'] for r in chosen};n=len(chosen)
    return {'eligible_races':total,'eligible_runners':len(rows),'selected_races':len(races),'selections':n,'losses':sum(1-r['y'] for r in chosen),'wins':sum(r['y'] for r in chosen),'abstention_races':total-len(races),'selection_race_fraction':len(races)/total,
        'mean_market':float(np.mean([r['market'] for r in chosen])) if n else None,'mean_model':float(np.mean([r['predictions']['selected_linear'] for r in chosen])) if n else None,'win_rate':float(np.mean([r['y'] for r in chosen])) if n else None,'odds_quantiles':np.quantile([r['odds'] for r in chosen],[0,.25,.5,.75,1]).tolist() if n else None}


def summarize(rows,out,ledger):
    from scripts.offline_systematic_search import starts,OUTER,rule_mask
    seed=20260924;rng=np.random.default_rng(seed);s,_=starts(rows);dates=np.array([rows[i]['race_date'] for i in s]);unique=sorted(set(dates));blocks=[np.flatnonzero(dates==d) for d in unique]
    names=list(rows[0]['predictions']);market=np.array([r['market'] for r in rows]);mll,mbs,*_=race_arrays(rows,market)
    overall={};deltas={};brier_deltas={}
    for name in names:
        p=np.array([r['predictions'][name] for r in rows]);ll,bs,*_=race_arrays(rows,p);deltas[name]=ll-mll;brier_deltas[name]=bs-mbs;overall[name]=metrics(rows,p)
    bootstrap={name:[] for name in names};brier_boot={name:[] for name in names}
    for _ in range(3000):
        idx=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        for name in names:bootstrap[name].append(deltas[name][idx].mean());brier_boot[name].append(brier_deltas[name][idx].mean())
    family=[name for name in names if name not in ['market','uniform']]
    estimates=np.array([deltas[name].mean() for name in family]);boot=np.array([bootstrap[name] for name in family]).T;sd=boot.std(0,ddof=1);sd[sd<1e-12]=1e-12
    maximum=np.max(np.abs((boot-estimates)/sd),axis=1);critical=float(np.quantile(maximum,.95))
    for name in names:
        overall[name].update({'delta_log_loss':float(deltas[name].mean()),'ll_date_ci95':np.quantile(bootstrap[name],[.025,.975]).tolist(),'delta_brier':float(brier_deltas[name].mean()),'brier_date_ci95':np.quantile(brier_boot[name],[.025,.975]).tolist(),'negative_dates':int(sum(deltas[name][dates==d].mean()<0 for d in unique)),'date_blocks':len(unique),'leave_one_date_out_delta_range':[float(min(deltas[name][dates!=d].mean() for d in unique)),float(max(deltas[name][dates!=d].mean() for d in unique))]})
        if name in family:
            j=family.index(name);overall[name]['simultaneous_reported_family_ci95']=[float(estimates[j]-critical*sd[j]),float(estimates[j]+critical*sd[j])]
    periods={};venues={};bands={}
    for name in names:
        periods[name]={}
        for outer in OUTER:
            subset=[r for r in rows if r['outer']==outer['name']];p=np.array([r['predictions'][name] for r in subset]);ref=np.array([r['market'] for r in subset]);periods[name][outer['name']]={**metrics(subset,p),'delta_log_loss':float((race_arrays(subset,p)[0]-race_arrays(subset,ref)[0]).mean())}
        venues[name]={}
        for venue in sorted({r['venue'] for r in rows}):
            subset=[r for r in rows if r['venue']==venue];p=np.array([r['predictions'][name] for r in subset]);ref=np.array([r['market'] for r in subset]);venues[name][venue]={'races':len(starts(subset)[0]),'delta_log_loss':float((race_arrays(subset,p)[0]-race_arrays(subset,ref)[0]).mean())}
        bands[name]=[]
        for lo,hi in [(1,3),(3,5),(5,10),(10,20),(20,10000)]:
            subset=[r for r in rows if lo<=r['odds']<hi];p=np.array([r['predictions'][name] for r in subset]);y=np.array([r['y'] for r in subset]);mp=np.array([r['market'] for r in subset])
            bands[name].append({'minimum':lo,'maximum_exclusive':hi,'runners':len(subset),'wins':int(y.sum()),'actual_rate':float(y.mean()),'market_mean':float(mp.mean()),'model_mean':float(p.mean()),'binary_brier_delta':float(np.mean((p-y)**2-(mp-y)**2))})
    selected={kind:{'overall':selection_stats(rows,flag),'periods':{o['name']:selection_stats([r for r in rows if r['outer']==o['name']],flag) for o in OUTER}} for kind,flag in [('favorite','favorite_selected'),('outsider','outsider_selected')]}
    # All frozen rule settings and nearby settings are descriptive only; no reselection.
    selections=json.loads((out/'frozen_selections.json').read_text());sensitivity=[]
    for selection in selections:
        subset=[r for r in rows if r['outer']==selection['outer']['name']];p=np.array([r['predictions']['selected_linear'] for r in subset])
        for kind,choice in selection['rules'].items():
            if not choice:continue
            chosen=choice['rule'];near=[max(0,chosen['threshold']-.005),chosen['threshold'],chosen['threshold']+.005]
            for threshold in sorted(set(near)):
                rule={**chosen,'threshold':threshold};mask=rule_mask(subset,p,rule);updated=[{**r,'selected':bool(m)} for r,m in zip(subset,mask)];sensitivity.append({'outer':selection['outer']['name'],'rule':rule,**selection_stats(updated,'selected')})
    write(out/'selection_sensitivity.json',sensitivity)
    # Every runner, including every nonselection, has its flags in the sealed file.
    write(out/'selection_results.json',selected);write(out/'period_metrics.json',periods);write(out/'venue_metrics.json',venues);write(out/'odds_band_calibration.json',bands)
    report={'overall':overall,'inference':{'bootstrap_repetitions':3000,'date_blocks':len(unique),'simultaneous_family':family,'critical_max_standardized_bootstrap_deviation':critical,'warning':'simultaneous intervals cover reported comparator family only; nested chronological selection reduces new search optimism but prior inspection and small date count preclude confirmation','no_untouched_holdout':True},'returns':'NOT_COMPUTED: retrospective latest-prejump retention selection, quote persistence/acceptance, deductions and final scratch timing unverified; largest-payout sensitivity not applicable'}
    write(out/'summary.json',report);ledger.append('OUTER_EVALUATION_COMPLETE',races=len(s),date_blocks=len(unique),models=len(names),confirmation=False)
    return report
