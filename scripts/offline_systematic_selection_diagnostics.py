"""Prespecified influence/sensitivity follow-up of sealed selection decisions."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.offline_prediction_research import write,sha
from scripts.offline_systematic_search import rule_mask
from scripts.offline_systematic_summary import selection_stats


def diagnose(out):
    rows=[json.loads(l) for l in (out/'outer_predictions.jsonl').open()]
    dates=sorted({r['race_date'] for r in rows});rng=np.random.default_rng(20260924)
    result={}
    for kind,flag in [('favorite','favorite_selected'),('outsider','outsider_selected')]:
        selected=[r for r in rows if r[flag]]
        def gap(rr):return float(np.mean([r['y']-r['market'] for r in rr])) if rr else None
        blocks=[[r for r in selected if r['race_date']==day] for day in dates];boot=[];empty=0
        for _ in range(3000):
            sample=[r for i in rng.integers(0,len(dates),len(dates)) for r in blocks[i]]
            if sample:boot.append(gap(sample))
            else:empty+=1
        influence={d:{'remaining_selections':len([r for r in selected if r['race_date']!=d]),'remaining_wins':sum(r['y'] for r in selected if r['race_date']!=d),'mean_actual_minus_market':gap([r for r in selected if r['race_date']!=d])} for d in dates}
        winner_remove=[]
        for winner in [r for r in selected if r['y']]:
            rr=[r for r in selected if r is not winner]
            winner_remove.append({'removed_race':winner['race_id'],'removed_box':winner['box'],'quoted_odds':winner['odds'],'remaining_selections':len(rr),'remaining_wins':sum(r['y'] for r in rr),'mean_actual_minus_market':gap(rr)})
        binary=np.mean([(r['y']-r['predictions']['selected_linear'])**2-(r['y']-r['market'])**2 for r in selected]) if selected else None
        result[kind]={'all_eligible_dates':len(dates),'selected_dates':len({r['race_date'] for r in selected}),'winning_dates':len({r['race_date'] for r in selected if r['y']}),'mean_actual_minus_market':gap(selected),'date_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist() if boot else None,'zero_selection_bootstraps':empty,'selected_binary_brier_delta_model_minus_market':float(binary) if binary is not None else None,'leave_one_date_out':influence,'leave_one_selected_win_out':winner_remove,'interpretation':'winner removal is predictive sensitivity, not a simulated payout or return'}
    settings=json.loads((out/'frozen_selections.json').read_text());neighbors=[]
    for setting in settings:
        name=setting['outer']['name'];rr=[r for r in rows if r['outer']==name];p=np.array([r['predictions']['selected_linear'] for r in rr])
        choice=setting['rules']['outsider']
        if not choice:continue
        for multiplier in [.8,1,1.2]:
            rule={**choice['rule'],'minimum_odds':choice['rule']['minimum_odds']*multiplier};mask=rule_mask(rr,p,rule)
            subset=[{**r,'flag':bool(m)} for r,m in zip(rr,mask)]
            neighbors.append({'outer':name,'rule':rule,'diagnostic_only_no_reselection':True,**selection_stats(subset,'flag')})
    write(out/'selection_influence.json',{'selections':result,'nearby_odds':neighbors,'prediction_sha256':sha(out/'outer_predictions.jsonl'),'status':'DIAGNOSTIC_ONLY_NO_REFIT_OR_RESELECTION','returns':'not estimated; no qualified executable quote history'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);diagnose(parser.parse_args().out)
