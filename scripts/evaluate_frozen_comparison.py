"""One terminal four-way evaluation; no collection and no interim result access.

A separately authorized closure receipt, fixed endpoint and immutable comparison
plan are required before even constructing the official-result reader.
"""
from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from src.predictor.future_comparison import MODELS, checked, load_plan, put, stamp, verify_comparison
from src.predictor.on_demand import canonical_bytes, verify_indexed_prediction_bundle


def summarize(races, *, replicates=20000):
    """Fixed metrics; paired race means and whole-date resampling. No fitting."""
    import numpy as np
    if not races: raise ValueError("no_complete_races")
    dates=sorted({r['date'] for r in races}); date_ix={d:i for i,d in enumerate(dates)}
    losses={}; metrics={}; calibration={}
    for model in MODELS:
        ll=[]; bs=[]; ranks=[]; bins=[[] for _ in range(10)]
        for race in races:
            p=np.asarray(race['probabilities'][model],dtype=float); y=np.zeros(len(p)); y[race['winner']]=1
            if not np.isfinite(p).all() or (p<=0).any() or not np.isclose(p.sum(),1,rtol=0,atol=1e-12):
                raise ValueError('invalid_probability')
            ll.append(float(-np.log(p[race['winner']])));bs.append(float(np.sum((p-y)**2)))
            order=sorted(range(len(p)),key=lambda i:(-p[i],race['boxes'][i]));ranks.append(order.index(race['winner'])+1)
            for probability,label in zip(p,y): bins[min(int(probability*10),9)].append((float(probability),float(label)))
        losses[model]=np.asarray([ll,bs]).T
        metrics[model]={'log_loss':float(np.mean(ll)),'brier':float(np.mean(bs)),'top_choice_accuracy':float(np.mean(np.asarray(ranks)==1)),
            'winner_mean_rank':float(np.mean(ranks)),'mean_reciprocal_rank':float(np.mean(1/np.asarray(ranks)))}
        calibration[model]=[{'lo':i/10,'hi':(i+1)/10,'runners':len(b),'predicted':float(np.mean([p for p,y in b])) if b else None,
            'observed':float(np.mean([y for p,y in b])) if b else None} for i,b in enumerate(bins)]
    counts=np.zeros(len(dates))
    for race in races: counts[date_ix[race['date']]]+=1
    rng=np.random.default_rng(20260924)
    draws=rng.multinomial(len(dates),np.full(len(dates),1/len(dates)),size=replicates)
    denominators=draws@counts
    paired={}
    # Four contrasts: each frozen candidate against each practical comparator.
    # Bonferroni 98.75% two-sided intervals per contrast. Requiring BOTH losses
    # to improve is an intersection-union rule within each contrast.
    for candidate in ('residual_box','residual_half'):
        for reference in ('market','production'):
            diff=losses[candidate]-losses[reference]; aggregate=np.zeros((len(dates),2))
            for row,race in zip(diff,races): aggregate[date_ix[race['date']]]+=row
            samples=(draws@aggregate)/denominators[:,None]
            bounds=np.quantile(samples,[.00625,.99375],axis=0)
            paired[candidate+'-minus-'+reference]={'log_loss':float(diff[:,0].mean()),'brier':float(diff[:,1].mean()),
                'simultaneous_family_intervals':bounds.T.tolist(),
                'both_upper_bounds_below_zero':bool((bounds[1]<0).all()),
                'leave_one_date_out_log_loss':[float((aggregate[:,0].sum()-a)/(len(races)-n)) for a,n in zip(aggregate[:,0],counts)] if len(dates)>1 else []}
    periods={}
    # Predeclared calendar-month/venue summaries, no selection or retuning.
    for kind in ('month','venue'):
        groups=sorted({r['date'][:7] if kind=='month' else r['venue'] for r in races})
        periods[kind]={g:{'races':sum((r['date'][:7] if kind=='month' else r['venue'])==g for r in races),
            'mean_losses':{m:losses[m][[(r['date'][:7] if kind=='month' else r['venue'])==g for r in races]].mean(axis=0).tolist() for m in MODELS}} for g in groups}
    return {'races':len(races),'dates':len(dates),'metrics':metrics,'paired':paired,'calibration':calibration,'periods':periods,
        'bootstrap':{'seed':20260924,'replicates':replicates,'unit':'Melbourne calendar date','contrasts':4,'family_alpha':.05},
        'interpretation':'No promotion; inspect coverage, closure and cross-date dependence before any claim.'}


def evaluate(plan_path, plan_sha, authority_path, authority_sha, result_database, out):
    now=datetime.now(timezone.utc)
    plan,_=load_plan(plan_path,plan_sha)
    if plan['status']!='AUTHORIZED': raise ValueError('real_evaluation_not_authorized')
    if now<stamp(plan['ends_at'])+timedelta(days=14): raise ValueError('fixed_result_closure_endpoint_not_reached')
    authority=json.loads(checked(authority_path,authority_sha))
    if authority.get('status')!='AUTHORIZED_ONE_SHOT_OUTCOMES' or authority.get('plan_sha256')!=plan_sha or not authority.get('authority_reference'):
        raise ValueError('outcome_authority_missing')
    programme=Path(plan['programme_root'])/plan_sha
    # Irrevocable global evaluation claim; failures stay consumed, no hidden retries.
    put(programme/'evaluation_claim.json',{'claimed_at':now.isoformat(),'authority_sha256':authority_sha,'output':str(out)})
    out.mkdir(parents=True,exist_ok=False)
    roots=[Path(p) for p in plan['prediction_output_roots']]
    opportunities=[json.loads(p.read_bytes()) for p in sorted((programme/'opportunities').glob('*.json'))]
    attempts=[]; common=[]
    for directory in sorted((programme/'attempts').glob('*')):
        admission=directory/'admission.json'; completion=directory/'completion.json'
        record={'attempt':directory.name,'admission':str(admission),'status':'MISSING_PREDICTION'}
        if admission.exists() and completion.exists():
            sealed=json.loads(completion.read_bytes()); matches=[r for r in roots if (r/sealed['bundle_entry']['directory']).is_dir()]
            if len(matches)!=1: raise ValueError('comparison_output_root_ambiguous')
            value=verify_comparison(matches[0],admission,expected_plan_sha256=plan_sha)
            record.update(status='COMMON_SEALED' if value['future_race_evidence'] else 'INCOMPLETE_COMPARISON',models=value['completion']['models'])
            if value['future_race_evidence']: common.append((matches[0],value))
        attempts.append(record)
    # Membership, missingness and exclusions frozen BEFORE any result lookup.
    put(out/'membership.json',{'plan_sha256':plan_sha,'opportunities':opportunities,'attempts':attempts,'common_races':len(common),'locked_at':now.isoformat()})
    from src.operator_ui.journal_results import OfficialResultSource
    source=OfficialResultSource(result_database); races=[]; closure=[]
    for root,value in common:
        record=value['records']['production']; race=record['race']
        if not stamp(plan['starts_at'])<=stamp(race['jump_timestamp'])<stamp(plan['ends_at']): raise ValueError('outside_allocated_population')
        verified=verify_indexed_prediction_bundle(root,value['completion']['bundle_entry'])
        inputs=json.loads((root/verified.directory/'comparison/inputs.json').read_bytes())
        job=SimpleNamespace(input=SimpleNamespace(race_id=race['race_id'],jump_timestamp=race['jump_timestamp'],
            ordered_runners=[{'box':r['box_number'],'name':r['display_name'],'source_native_runner_id':r.get('source_native_runner_id')} for r in inputs['runners']]))
        result=source.read(job,verified,now=now)
        closure.append({'race_id':race['race_id'],**result})
        if result['state']!='RESULT_AVAILABLE': continue
        winner=next(r['box_number'] for r in result['evidence']['runner_rows'] if r['is_winner'])
        boxes=[r['box_number'] for r in record['predictions']]
        races.append({'race_id':race['race_id'],'date':race['race_date'],'venue':race['venue'],'boxes':boxes,'winner':boxes.index(winner),
            'probabilities':{m:[r['probability'] for r in value['records'][m]['predictions']] for m in MODELS}})
    put(out/'result_closure.json',closure)
    if len(races)!=len(common) or not common:
        put(out/'assessment.json',{'status':'INCOMPLETE_RESULT_CLOSURE_NO_CONFIRMATORY_CLAIM','common_seals':len(common),'complete_results':len(races),
            'descriptive_complete_results_only':summarize(races) if races else None})
        return
    put(out/'assessment.json',summarize(races))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','authority','result-database','out'): p.add_argument('--'+name,type=Path,required=True)
    for name in ('plan-sha256','authority-sha256'): p.add_argument('--'+name,required=True)
    a=p.parse_args();evaluate(a.plan,a.plan_sha256,a.authority,a.authority_sha256,a.result_database,a.out)
