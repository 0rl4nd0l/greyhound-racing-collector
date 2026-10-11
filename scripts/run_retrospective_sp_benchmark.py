#!/usr/bin/env python3
"""Matched result-SP diagnostic. Never a decision-time or executable-price test.

Only exact development package keys may cause raw-result bodies to be opened.
The source manifest, reserved metadata and artifact hashes are read first.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from scipy.optimize import minimize
from bs4 import BeautifulSoup

SOURCE = Path('/mnt/tenn-nvme2/tenn/greyhound-historical-improvement-execution-20261010-evidence')
CORPUS = Path('/mnt/tenn-nvme2/tenn/greyhound-historical-csv-expansion-20261008-evidence/dataset-03/manifest.json')
JOINED = Path('/mnt/tenn-nvme2/tenn/greyhound-historical-csv-expansion-20261008-evidence/http-recovery-20261009/run-01/prepared/000084/joined')
STREAMS = {'hybrid_sp_combo': 'matched_hybrid65', 'tree_sp_combo': 'saved_tree95'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def lines(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def verify_artifacts(directory):
    manifest=json.loads((directory/'artifacts.sha256.json').read_text())
    # These frozen source manifests map relative filenames to digests.
    for name, digest in manifest.items():
        if isinstance(digest, dict):
            digest=digest['sha256']
        if sha(directory/name)!=digest:
            raise ValueError(f'ARTIFACT_HASH_MISMATCH:{name}')


def parse_sp(raw, expected):
    """Require exact native identity and guide box; never infer missing prices."""
    table=re.findall(rb'<table\b[^>]*class="[^"]*race-runners--result[^\"]*"[^>]*>.*?</table>',raw,re.S)
    if len(table)!=1:
        raise ValueError('RESULT_TABLE_CARDINALITY')
    soup=BeautifulSoup(table[0], 'html.parser')
    observed={}
    for row in soup.select('tr.race-runner'):
        boxel=row.select_one('td.race-runners__box sprite-svg')
        dogel=row.select_one('blackbook-dog[data-dog-id]')
        if boxel is None or dogel is None:
            raise ValueError('NATIVE_IDENTITY_MISSING')
        match=re.fullmatch(r'rug_(\d+)', str(boxel.get('name','')))
        if not match: raise ValueError('BOX_INVALID')
        box=int(match[1]); native=str(dogel['data-dog-id'])
        if box in observed: raise ValueError('DUPLICATE_BOX')
        cell=row.select_one('td.race-runners__starting-price')
        token=cell.get_text(' ',strip=True) if cell else ''
        matched=re.fullmatch(r'\$?(\d+(?:\.\d+)?)', token)
        price=float(matched[1]) if matched else None
        observed[box]=(native,price,token)
    result=[]
    for box,native in expected:
        if box not in observed or observed[box][0]!=native:
            raise ValueError('NATIVE_RUNNER_MISMATCH')
        price=observed[box][1]
        if price is None or not math.isfinite(price) or price<=1:
            raise ValueError('MISSING_OR_INVALID_ACTIVE_SP')
        result.append(price)
    return result


def softmax(score):
    score=score-np.max(score)
    result=np.exp(score)
    return result/result.sum()


def metrics(races, model):
    rows=[]; bins=[{'n':0,'sum_p':0.,'wins':0.} for _ in range(10)]
    for r in races:
        p=np.asarray([v['probabilities'][model] for v in r['runners']]); y=np.asarray([v['y'] for v in r['runners']]); win=int(np.argmax(y))
        top=np.flatnonzero(p==p.max()); credit=float(y[top].sum()/len(top))
        rows.append({'race':r['source_race_key'],'date':r['race_date'],'track':r['track'],'log_loss':float(-np.log(p[win])),'brier':float(np.square(p-y).sum()),'top1':credit})
        for q,v in zip(p,y):
            b=bins[min(9,int(q*10))];b['n']+=1;b['sum_p']+=float(q);b['wins']+=float(v)
    return {'races':len(rows),'log_loss':float(np.mean([x['log_loss'] for x in rows])),'brier':float(np.mean([x['brier'] for x in rows])),'top1':float(np.mean([x['top1'] for x in rows])),'winner_credit':sum(x['top1'] for x in rows),'calibration_bins':bins},rows


def fit_correction(races, stream=None):
    """One shared market slope; form adds a shrunken bounded coefficient.

    Objective is equal-race NLL + .01*((a-1)^2+b^2).
    Common market slope a in [.5,1.5]; form correction b in [0,.5].
    q = softmax(a log(p_market)+b log(p_form)).
    """
    arrays=[]
    for r in races:
        market=np.log([v['probabilities']['reported_sp'] for v in r['runners']])
        form=np.log([v['probabilities'][stream] for v in r['runners']]) if stream else np.zeros(len(market))
        arrays.append((np.column_stack([market,form])[:,:2 if stream else 1],np.asarray([v['y'] for v in r['runners']])))
    anchor=np.array([1.,0.] if stream else [1.])
    def objective(beta):
        loss=0.; grad=np.zeros(len(beta))
        for x,y in arrays:
            p=softmax(x@beta);loss-=float(np.dot(y,np.log(p)));grad+=x.T@(p-y)
        penalty=0.01*np.square(beta-anchor).sum()
        return loss/len(arrays)+penalty,grad/len(arrays)+.02*(beta-anchor)
    result=minimize(objective,anchor,jac=True,method='L-BFGS-B',bounds=[(.5,1.5)]+([(0,.5)] if stream else []),options={'ftol':1e-12,'gtol':1e-8,'maxiter':200})
    if not result.success: raise ValueError(f'OPTIMIZATION_FAILED:{result.message}')
    return {'coefficients':result.x.tolist(),'success':bool(result.success),'objective':float(result.fun),'iterations':int(result.nit),'train_races':len(races),'train_dates':sorted({r['race_date'] for r in races}),'form_stream':stream}


def apply(races, model, fit):
    a=fit['coefficients'][0]; b=fit['coefficients'][1] if fit['form_stream'] else 0
    for r in races:
        score=a*np.log([v['probabilities']['reported_sp'] for v in r['runners']])
        if fit['form_stream']:score+=b*np.log([v['probabilities'][fit['form_stream']] for v in r['runners']])
        for v,p in zip(r['runners'],softmax(score)):v['probabilities'][model]=float(p)


def prepare(out):
    if (out/'matched_sp.jsonl').exists():
        raise ValueError('PREPARATION_ALREADY_EXISTS')
    corpus=json.loads(CORPUS.read_text()); manifest=json.loads((JOINED/'results_manifest.json').read_text())
    if manifest['allowed_label_splits']!=['train','validation']:raise ValueError('ACCESS_SPLIT_CONTRACT')
    reserved={r['source_race_key'] for r in corpus['races'] if r['proposed_split']=='test'}
    assert len(reserved)==169
    verify_artifacts(SOURCE/'feature-package-01');verify_artifacts(SOURCE/'run-01')
    mappings=defaultdict(dict)
    for row in lines(JOINED/'runner_mapping.jsonl'):
        if row['source_race_key'] in reserved:raise ValueError('RESERVED_MAPPING')
        mappings[row['source_race_key']][row['guide_box']]=row
    refs={r['source_race_key']:r for r in manifest['results']}
    allowed=set(manifest['allowed_source_race_keys'])
    exclusions=[];pop=[]; counts={}; memberships=[]
    for split,count in [('train',720),('development',656),('later',975)]:
        groups=lines(SOURCE/'feature-package-01'/f'{split}.jsonl');assert len(groups)==count
        predictions={(r['source_race_key'],r['guide_box']):r['probabilities'] for r in lines(SOURCE/'run-01'/f'{split}_predictions.jsonl')}
        counts[split]={'source_races':count,'eligible_sp':0,'excluded':0,'dates':len({r['metadata']['race_date'] for r in groups})}
        for group in groups:
            key=group['metadata']['source_race_key'];memberships.append({'source_race_key':key,'partition':split})
            if key not in allowed or key in reserved:raise ValueError('BODY_ACCESS_DENIED')
            ref=refs[key];raw=Path(ref['raw_html']['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=ref['raw_html']['sha256'] or ref['raw_html']['sha256']!=group['metadata']['raw_response_sha256']:raise ValueError('BODY_HASH_MISMATCH')
            boxes=[r['metadata']['guide_box'] for r in group['runners']]
            expected=[(box,mappings[key][box]['source_native_dog_id']) for box in boxes]
            try:prices=parse_sp(raw,expected)
            except ValueError as e:
                exclusions.append({'source_race_key':key,'partition':split,'reason':str(e),'body':ref['raw_html']});counts[split]['excluded']+=1;continue
            inv=np.reciprocal(prices);market=inv/inv.sum()
            rows=[]
            for runner,box,price,p in zip(group['runners'],boxes,prices,market):
                probs=dict(predictions[(key,box)]);probs['reported_sp']=float(p)
                rows.append({'guide_box':box,'final_box':runner['metadata']['final_box'],'native_dog_id':mappings[key][box]['source_native_dog_id'],'y':int(runner['target']['is_winner']),'sp':price,'probabilities':probs})
            assert sum(r['y'] for r in rows)==1
            pop.append({'source_race_key':key,'race_date':group['metadata']['race_date'],'track':key.split(' - ')[1],'partition':split,'overround':float(inv.sum()),'body':ref['raw_html'],'runners':rows})
            counts[split]['eligible_sp']+=1
    dump(out/'eligibility.json',{'coverage':counts,'reserved_test_races':len(reserved),'protected_bodies_opened':0,'decision_time_market_races':0,'incumbent_eligible_races':0,'incumbent_exclusion':'July2026 training postdates2025 targets; installed feature/DB route absent for historical2025 inputs','source_manifest_sha256':sha(JOINED/'results_manifest.json'),'corpus_manifest_sha256':sha(CORPUS),'source_feature_manifest_sha256':sha(SOURCE/'feature-package-01/artifacts.sha256.json'),'source_prediction_manifest_sha256':sha(SOURCE/'run-01/artifacts.sha256.json')})
    dump(out/'memberships.json',memberships);dump(out/'exclusions.json',exclusions)
    with (out/'matched_sp.jsonl').open('w') as f:
        for r in pop:f.write(json.dumps(r,sort_keys=True)+'\n')


def run(out):
    races=lines(out/'matched_sp.jsonl');development=[r for r in races if r['partition']=='development'];later=[r for r in races if r['partition']=='later']
    streams={'market_calibrated':None,**STREAMS};dates=sorted({r['race_date'] for r in development})
    protocol={'status':'RECORDED_BEFORE_META_FITS','base_training_end':'2025-08-31','calibration_dates':dates,'diagnostic_dates':sorted({r['race_date'] for r in later}),'form_streams':streams,'tree_rule':'Correction uses uncalibrated tree95, never development-fitted temperature','penalty':'.01*((market_slope-1)^2+form_coefficient^2)','bounds':{'market_slope':[.5,1.5],'form_coefficient':[0,.5]},'folds':[{'train_dates':dates[:i],'test_date':date} for i,date in enumerate(dates) if i>0],'fit_budget':len(streams)*len(dates),'selection_or_tuning':False,'market_semantics':'Post-result source-reported SP; no historical capture cutoff or executable quote assertion'}
    if (out/'protocol.json').exists():raise ValueError('OUTPUT_ALREADY_CONSUMED')
    dump(out/'protocol.json',protocol)
    dump(out/'executing_source.json',{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)})
    fits=[]
    for i,date in enumerate(dates):
        if not i:continue
        train=[r for r in development if r['race_date']<date];test=[r for r in development if r['race_date']==date]
        for name,stream in streams.items():
            intent={'model':name,'kind':'prequential','test_date':date,'train_dates':dates[:i]};dump(out/f'fit-intent-{len(fits):02d}.json',intent)
            fit=fit_correction(train,stream);fits.append({**intent,**fit});apply(test,name,fit)
            dump(out/f'fit-{len(fits)-1:02d}.json',fits[-1])
    for name,stream in streams.items():
        intent={'model':name,'kind':'final','test_dates':protocol['diagnostic_dates'],'train_dates':dates};dump(out/f'fit-intent-{len(fits):02d}.json',intent)
        fit=fit_correction(development,stream);fits.append({**intent,**fit});apply(later,name,fit);dump(out/f'fit-{len(fits)-1:02d}.json',fits[-1])
    scores={};race_scores=[]
    base_models=['reported_sp','matched_hybrid65','saved_tree95','saved_tree95_temperature']
    for partition,rs in [('development_prequential',[r for r in development if r['race_date']>dates[0]]),('later',later)]:
        scores[partition]={}
        for model in base_models+list(streams):
            if partition=='development_prequential' and model=='saved_tree95_temperature':continue
            metric,rows=metrics(rs,model);scores[partition][model]=metric
            race_scores.extend({**r,'partition':partition,'model':model} for r in rows)
    dump(out/'summary.json',{'metrics':scores,'fit_count':len(fits),'final_fits':fits[-len(streams):],'inference':'exploratory reused outcomes,10 later dates; result reconstructed roster and post-result SP','confirmed_improvement':False})
    dump(out/'race_scores.json',race_scores)
    with (out/'predictions.jsonl').open('w') as f:
        for r in development+later:f.write(json.dumps(r,sort_keys=True)+'\n')
    print(json.dumps({p:{m:{k:v for k,v in x.items() if k!='calibration_bins'} for m,x in ms.items()} for p,ms in scores.items()},indent=2))


def add_dynamic(out, base, dynamic):
    """Run only the predeclared recency/dynamic meta arms, preserving base fits."""
    if (out/'protocol.json').exists():
        raise ValueError('OUTPUT_ALREADY_CONSUMED')
    races=lines(base/'predictions.jsonl')
    verify_artifacts(dynamic)
    sources={}
    for partition in ['development','later']:
        for row in lines(dynamic/f'{partition}_predictions.jsonl'):
            key=(row['source_race_key'],row['guide_box'])
            if key in sources:raise ValueError('DUPLICATE_DYNAMIC_PREDICTION')
            sources[key]=row
    observed=set()
    for race in races:
        for runner in race['runners']:
            key=(race['source_race_key'],runner['guide_box']);row=sources[key]
            if row['race_date']!=race['race_date']:raise ValueError('DYNAMIC_DATE_MISMATCH')
            observed.add(key)
            for stream in ['recency','dynamic']:
                p=row['probabilities'][stream]
                if not 0<p<1:raise ValueError('DYNAMIC_PROBABILITY_INVALID')
                runner['probabilities'][stream]=p
        for stream in ['recency','dynamic']:
            if abs(sum(v['probabilities'][stream] for v in race['runners'])-1)>1e-10:
                raise ValueError('DYNAMIC_FIELD_MISMATCH')
    if observed!=set(sources):raise ValueError('DYNAMIC_MEMBERSHIP_MISMATCH')
    development=[r for r in races if r['partition']=='development'];later=[r for r in races if r['partition']=='later']
    dates=sorted({r['race_date'] for r in development})
    streams={'recency_sp_combo':'recency','dynamic_sp_combo':'dynamic'}
    protocol={'status':'RECORDED_BEFORE_META_FITS','base_protocol':{'path':str(base/'protocol.json'),'sha256':sha(base/'protocol.json')},'dynamic_artifacts':{'path':str(dynamic/'artifacts.sha256.json'),'sha256':sha(dynamic/'artifacts.sha256.json')},'streams':streams,'folds':[{'train_dates':dates[:i],'test_date':date} for i,date in enumerate(dates) if i>0],'fit_budget':len(dates)*2,'base_prediction_train_end':'2025-08-31','bounds':{'market_slope':[.5,1.5],'form_coefficient':[0,.5]},'penalty':'.01*((market_slope-1)^2+form_coefficient^2)','selection_or_tuning':False}
    dump(out/'protocol.json',protocol);dump(out/'executing_source.json',{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)});fits=[]
    for i,date in enumerate(dates):
        if not i:continue
        train=[r for r in development if r['race_date']<date];test=[r for r in development if r['race_date']==date]
        for name,stream in streams.items():
            intent={'model':name,'kind':'prequential','test_date':date,'train_dates':dates[:i]};dump(out/f'fit-intent-{len(fits):02d}.json',intent)
            fit=fit_correction(train,stream);fits.append({**intent,**fit});apply(test,name,fit);dump(out/f'fit-{len(fits)-1:02d}.json',fits[-1])
    for name,stream in streams.items():
        intent={'model':name,'kind':'final','train_dates':dates};dump(out/f'fit-intent-{len(fits):02d}.json',intent)
        fit=fit_correction(development,stream);fits.append({**intent,**fit});apply(later,name,fit);dump(out/f'fit-{len(fits)-1:02d}.json',fits[-1])
    summary={};scores=[]
    for partition,rs in [('development_prequential',[r for r in development if r['race_date']>dates[0]]),('later',later)]:
        summary[partition]={}
        models=['reported_sp','market_calibrated','matched_hybrid65','hybrid_sp_combo','tree_sp_combo','recency','dynamic',*streams]
        if partition=='later':models.append('saved_tree95_temperature')
        for model in models:
            metric,rows=metrics(rs,model);summary[partition][model]=metric;scores.extend({**r,'partition':partition,'model':model} for r in rows)
    dump(out/'summary.json',{'metrics':summary,'additional_fit_count':len(fits),'final_fits':fits[-2:],'confirmed_improvement':False})
    dump(out/'race_scores.json',scores)
    with (out/'predictions.jsonl').open('w') as f:
        for race in races:f.write(json.dumps(race,sort_keys=True)+'\n')
    print(json.dumps({'final_fits':fits[-2:],'later':{m:{k:v for k,v in x.items() if k!='calibration_bins'} for m,x in summary['later'].items()}},indent=2))


def seal(out):
    dump(out/'execution_source.json',{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)})
    dump(out/'artifacts.sha256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifacts.sha256.json'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--prepare',action='store_true');parser.add_argument('--fit',action='store_true');parser.add_argument('--base',type=Path);parser.add_argument('--dynamic',type=Path);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.prepare:prepare(args.output)
    if args.fit:run(args.output)
    if args.dynamic:
        if not args.base:raise ValueError('BASE_OUTPUT_REQUIRED')
        add_dynamic(args.output,args.base,args.dynamic)
    if args.fit or args.dynamic:seal(args.output)


if __name__=='__main__':main()
