"""Finite nested chronological search with an append-only, hash-chained ledger.

Only the prepared eligible development file is opened. All old outer dates are
acknowledged development. Outer labels are scored after all selections freeze.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, GradientBoostingRegressor
from scripts.offline_prediction_research import FROZEN_FEATURES, FOLDS, sha, write, scalar
from scripts.offline_systematic_features import GROUPS

SEED=20260924
OUTER=[
    {**FOLDS[0],'inner':[('2026-06-13','2026-06-14','2026-06-17'),('2026-06-17','2026-06-18','2026-06-21')]},
    {**FOLDS[1],'inner':[('2026-06-17','2026-06-18','2026-06-21'),('2026-06-21','2026-06-24','2026-06-30')]},
    {**FOLDS[2],'inner':[('2026-06-17','2026-06-18','2026-06-21'),('2026-06-21','2026-06-24','2026-06-30'),('2026-06-30','2026-07-01','2026-07-02')]},
]
RECIPES={'base16':list(FROZEN_FEATURES),'base_box':list(FROZEN_FEATURES)+['box_number']}
for name,features in GROUPS.items():
    RECIPES['add_'+name]=list(FROZEN_FEATURES)+features
RECIPES.update({
    'remove_recent':[f for f in FROZEN_FEATURES if not f.startswith('recent_')],
    'remove_long':[f for f in FROZEN_FEATURES if not f.startswith('career_')],
    'remove_context':[f for f in FROZEN_FEATURES if 'same_' not in f],
    'recent_only':[f for f in FROZEN_FEATURES if f.startswith('recent_') or f in ['prior_start_count','days_since_last_start']],
    'draw_times':list(FROZEN_FEATURES)+GROUPS['draw']+GROUPS['times'],
    'section_pressure':list(FROZEN_FEATURES)+GROUPS['sectionals']+GROUPS['pace_pressure'],
    'margin_recency':list(FROZEN_FEATURES)+GROUPS['margin_improvement']+GROUPS['recency'],
})
TREE_CONFIGS=[
    {'family':'hist','leaves':3,'iterations':60,'min_leaf':30},
    {'family':'hist','leaves':7,'iterations':80,'min_leaf':30},
    {'family':'extra','depth':3,'iterations':100,'min_leaf':20},
    {'family':'extra','depth':5,'iterations':100,'min_leaf':30},
    {'family':'residual_tree','depth':1,'iterations':80,'min_leaf':30},
    {'family':'residual_tree','depth':2,'iterations':60,'min_leaf':30},
]


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


class Ledger:
    def __init__(self,path):
        self.path=path; self.previous='0'*64; self.sequence=0
        if path.exists():
            raise ValueError('ledger already exists; no silent resume or overwrite')
    def append(self,event,**payload):
        row={'sequence':self.sequence,'event':event,'previous_sha256':self.previous,'seed':SEED,**payload}
        digest=hashlib.sha256(canonical(row).encode()).hexdigest()
        with self.path.open('a') as f:
            f.write(canonical({**row,'sha256':digest})+'\n');f.flush();os.fsync(f.fileno())
        self.previous=digest;self.sequence+=1


def starts(rows):
    if not rows: raise ValueError('empty race population')
    ids=[r['race_id'] for r in rows]
    boundaries=np.r_[0,np.flatnonzero(np.array(ids[1:])!=np.array(ids[:-1]))+1]
    if len(boundaries)!=len(set(ids)):raise ValueError('race rows must be contiguous')
    return boundaries,np.diff(np.r_[boundaries,len(rows)])


def softmax(rows,logits):
    s,n=starts(rows); shifted=logits-np.repeat(np.maximum.reduceat(logits,s),n)
    exp=np.exp(shifted);return exp/np.repeat(np.add.reduceat(exp,s),n)


def validate(rows,p):
    s,n=starts(rows); y=np.array([r['y'] for r in rows])
    if not np.all(np.isin(y,[0,1])) or not np.all(np.add.reduceat(y,s)==1):raise ValueError('single winner required')
    if len(p)!=len(rows) or np.any(~np.isfinite(p)) or np.any(p<=0):raise ValueError('invalid probabilities')
    if not np.allclose(np.add.reduceat(p,s),1,atol=1e-10,rtol=0):raise ValueError('probabilities do not sum to one')


def loss(rows,p):
    validate(rows,p);return float(-np.log(p[np.array([r['y'] for r in rows])==1]).mean())


def values(rows,names):
    result=[]
    for r in rows:
        vals=[]
        for name in names:
            if name.startswith('venue_box::'):
                value=r['features']['box_fraction'] if r['layout_venue']==name.split('::')[1] else 0.0
            else:value=r['features'].get(name)
            vals.append(np.nan if value is None else value)
        result.append(vals)
    return np.asarray(result,float)


def prep_fit(rows,names,center=True):
    names=list(dict.fromkeys(names))
    if '__box_track__' in names:
        names.remove('__box_track__')
        # Categories are learned from training only; later unseen tracks map zero.
        names += ['venue_box::'+v for v in sorted({r['layout_venue'] for r in rows})]
    x=values(rows,names)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x.T])
    expanded=np.c_[np.where(np.isfinite(x),x,med),~np.isfinite(x)]
    mu=expanded.mean(0); scale=expanded.std(0);scale[scale<1e-12]=1
    return {'names':names,'median':med,'mean':mu,'scale':scale,'center':center}


def transform(rows,prep):
    x=values(rows,prep['names']);x=(np.c_[np.where(np.isfinite(x),x,prep['median']),~np.isfinite(x)]-prep['mean'])/prep['scale']
    if prep['center']:
        s,n=starts(rows);x-=np.repeat(np.add.reduceat(x,s,axis=0)/n[:,None],n,axis=0)
    return x


def linear_fit(rows,names,l2):
    prep=prep_fit(rows,names);x=transform(rows,prep);s,n=starts(rows);nr=len(s)
    y=np.array([r['y'] for r in rows]);offset=np.log([r['market'] for r in rows])
    def objective(beta):
        z=x@beta; t=np.tanh(z/.35);p=softmax(rows,offset+.35*t)
        return -np.log(p[y==1]).mean()+.5*l2*(beta@beta), x.T@((p-y)*(1-t*t))/nr+l2*beta
    fitted=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12,'gtol':1e-8,'maxls':50})
    if not fitted.success:raise RuntimeError(str(fitted.message))
    return {'prep':prep,'beta':fitted.x,'kind':'linear','l2':l2,'iterations':int(fitted.nit)}


def fit_tree(rows,names,config):
    prep=prep_fit(rows,names,center=False);x=transform(rows,prep);y=np.array([r['y'] for r in rows]);market=np.array([r['market'] for r in rows])
    if config['family']=='hist':
        estimator=HistGradientBoostingClassifier(max_leaf_nodes=config['leaves'],max_iter=config['iterations'],min_samples_leaf=config['min_leaf'],learning_rate=.05,l2_regularization=10,early_stopping=False,random_state=SEED)
    elif config['family']=='extra':
        estimator=ExtraTreesClassifier(n_estimators=config['iterations'],max_depth=config['depth'],min_samples_leaf=config['min_leaf'],n_jobs=1,random_state=SEED)
    else:
        estimator=GradientBoostingRegressor(n_estimators=config['iterations'],max_depth=config['depth'],min_samples_leaf=config['min_leaf'],learning_rate=.05,loss='squared_error',random_state=SEED)
        y=y-market
    estimator.fit(x,y)
    return {'prep':prep,'estimator':estimator,'kind':config['family'],'config':config}


def predict(rows,model,strength=1):
    x=transform(rows,model['prep']);offset=np.log([r['market'] for r in rows])
    if model['kind']=='linear':
        return softmax(rows,offset+strength*.35*np.tanh(x@model['beta']/.35))
    if model['kind']=='residual_tree':
        return softmax(rows,offset+.35*np.tanh(model['estimator'].predict(x)/.35))
    raw=model['estimator'].predict_proba(x)[:,1]
    return softmax(rows,np.log(np.clip(raw,1e-12,1)))


def serialize_model(model):
    result={k:v for k,v in model.items() if k!='estimator'}
    def convert(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
        return value
    return convert(result)


def inner_splits(data,outer):
    result=[]
    for train_end,val_start,val_end in outer['inner']:
        train=[r for r in data if r['race_date']<=train_end]
        val=[r for r in data if val_start<=r['race_date']<=val_end]
        assert train_end<val_start<=val_end<outer['test_start']
        assert not {r['race_id'] for r in train}&{r['race_id'] for r in val}
        result.append((train,val))
    return result


class Search:
    def __init__(self,out,data):
        self.out=out;self.data=data;self.ledger=Ledger(out/'experiment_ledger.jsonl');self.fit_count=0;self.trial_count=0
        self.started=time.monotonic();self.cpu_started=time.process_time()
        self.ledger.append('PROGRAMME_START',input_sha256=sha(out.parent/'features/features.jsonl'),runner_sha256=sha(Path(__file__)),exposure='all previous evaluation dates are development')
    def stage(self,name,budget):
        self.stage_name=name;self.stage_cpu=time.process_time();self.budget=budget
        self.ledger.append('STAGE_START',stage=name,cpu_budget_seconds=budget)
    def check_budget(self):
        if time.process_time()-self.stage_cpu>self.budget:raise RuntimeError('stage CPU budget exhausted')
    def fitted(self,train,names,config,where):
        self.check_budget();before=time.process_time();wall=time.monotonic();self.fit_count+=1
        identity={'fit':self.fit_count,'stage':self.stage_name,'where':where,'config':config,'features':names,'train_races':len(starts(train)[0]),'train_min':min(r['race_date'] for r in train),'train_max':max(r['race_date'] for r in train),'train_identity_sha256':hashlib.sha256(canonical([(r['race_id'],r['box']) for r in train]).encode()).hexdigest()}
        self.ledger.append('FIT_START',**identity)
        try:
            model=linear_fit(train,names,config['l2']) if config['family']=='linear' else fit_tree(train,names,config)
        except Exception as exc:
            self.ledger.append('FIT_FAILED',**identity,error=str(exc),cpu_seconds=time.process_time()-before);raise
        self.ledger.append('FIT_COMPLETE',**identity,cpu_seconds=time.process_time()-before,wall_seconds=time.monotonic()-wall)
        return model
    def trial(self,outer,name,config,rows,p,question):
        self.trial_count+=1;score=loss(rows,p)
        path=self.out/'inner_predictions'/f'{outer}_{self.trial_count:04d}.npz';np.savez_compressed(path,probability=p)
        self.ledger.append('VALIDATION_TRIAL',trial=self.trial_count,stage=self.stage_name,outer=outer,name=name,config=config,question=question,log_loss=score,races=len(starts(rows)[0]),prediction_path=str(path.relative_to(self.out)),prediction_sha256=sha(path))
        return {'name':name,'config':config,'loss':score,'p':p,'trial':self.trial_count}
    def oof(self,splits,names,config,cache,key):
        if key not in cache:
            models=[self.fitted(train,names,config,f'inner{i}') for i,(train,_val) in enumerate(splits)]
            cache[key]=models
        return cache[key]
    def run(self):
        all_outer=[]; selections=[]; summaries=[]
        for outer in OUTER:
            oid=outer['name'];splits=inner_splits(self.data,outer);validation=[r for _,v in splits for r in v];market=np.array([r['market'] for r in validation]);cache={}
            identity=[{k:r[k] for k in ['race_id','race_date','box']} for r in validation]
            write(self.out/f'{oid}_inner_identities.json',identity)
            self.stage(oid+'/groups',180)
            group_trials=[]
            for name,names in RECIPES.items():
                models=self.oof(splits,names,{'family':'linear','l2':1.0},cache,(name,1.0))
                p=np.concatenate([predict(val,m) for m,(_,val) in zip(models,splits)])
                group_trials.append(self.trial(oid,name,{'recipe':name,'family':'linear','l2':1.0,'strength':1.0},validation,p,'single group increment, removal or fixed mechanism combination'))
            best_groups=sorted(group_trials,key=lambda t:(t['loss'],len(RECIPES[t['name']]),t['name']))[:3]
            self.ledger.append('GROUP_ELIMINATION',outer=oid,retained=[t['name'] for t in best_groups],eliminated=[t['name'] for t in group_trials if t not in best_groups],decision='retain best3 inner OOF log losses; outer labels unused')
            self.stage(oid+'/tuning',300)
            tuned=[]
            for g in best_groups:
                recipe=g['name']
                for l2 in [.1,.3,1.,3.]:
                    models=self.oof(splits,RECIPES[recipe],{'family':'linear','l2':l2},cache,(recipe,l2))
                    for strength in [.25,.5,1.]:
                        p=np.concatenate([predict(val,m,strength) for m,(_,val) in zip(models,splits)])
                        tuned.append(self.trial(oid,recipe,{'family':'linear','recipe':recipe,'l2':l2,'strength':strength},validation,p,'coarse penalty and shrinkage on retained feature recipes'))
            for power in [.75,1.,1.25,1.5]:
                p=softmax(validation,power*np.log(market))
                tuned.append(self.trial(oid,'market_power',{'family':'power','power':power},validation,p,'market calibration without form'))
            chosen=min(tuned,key=lambda t:(t['loss'],0 if t['config']['family']=='power' else 1,t['trial']))
            calibration=min([t for t in tuned if t['config']['family']=='power'],key=lambda t:t['loss'])
            # Nonlinear features are selected using only earlier OOF group screening.
            nonlinear_names=list(dict.fromkeys(RECIPES[best_groups[0]['name']]+GROUPS['market_shape']))
            self.stage(oid+'/nonlinear',120)
            trees=[]
            for i,config in enumerate(TREE_CONFIGS):
                models=self.oof(splits,nonlinear_names,config,cache,('tree',i))
                p=np.concatenate([predict(val,m) for m,(_,val) in zip(models,splits)])
                trial=self.trial(oid,f'tree{i}',{**config,'tree_index':i},validation,p,'limited nonlinear response to market and screened form')
                trees.append(trial)
            tree=min(trees,key=lambda t:(t['loss'],t['trial']))
            blends=[]
            for weight in [0,.25,.5,1.]:
                p=(1-weight)*chosen['p']+weight*tree['p']
                blends.append(self.trial(oid,'ensemble',{'tree_weight':weight},validation,p,'blend independently fitted OOF component probabilities; no in-sample stacking'))
            blend=min(blends,key=lambda t:(t['loss'],t['config']['tree_weight']))
            rules=select_rules(validation,chosen['p'],self.ledger,oid)
            selected={'outer':outer,'linear':{k:v for k,v in chosen.items() if k!='p'},'calibration':{k:v for k,v in calibration.items() if k!='p'},'tree':{k:v for k,v in tree.items() if k!='p'},'ensemble':{k:v for k,v in blend.items() if k!='p'},'nonlinear_features':nonlinear_names,'rules':rules,'group_ranking':[{k:v for k,v in t.items() if k!='p'} for t in sorted(group_trials,key=lambda t:t['loss'])],'inner_races':len(starts(validation)[0]),'inner_dates':len({r['race_date'] for r in validation})}
            self.ledger.append('OUTER_SELECTION_FROZEN',**selected)
            selections.append(selected)
            # Refit on all earlier dates, including inner validation; no outer labels consumed here.
            self.stage(oid+'/outer_fit',90)
            train=[r for r in self.data if r['race_date']<outer['test_start']]
            test=[r for r in self.data if outer['test_start']<=r['race_date']<=outer['test_end']]
            m=np.array([r['market'] for r in test]);pred={'market':m,'uniform':softmax(test,np.zeros(len(test)))}
            config=chosen['config'];latencies={}
            if config['family']=='power':pred['selected_linear']=softmax(test,config['power']*np.log(m)); selected_model=None
            else:
                selected_model=self.fitted(train,RECIPES[config['recipe']],config,oid+'/selected_linear')
                before=time.perf_counter();pred['selected_linear']=predict(test,selected_model,config['strength']);latencies['selected_linear_seconds']=time.perf_counter()-before
            pred['selected_market_calibration']=softmax(test,calibration['config']['power']*np.log(m))
            tmodel=self.fitted(train,nonlinear_names,TREE_CONFIGS[tree['config']['tree_index']],oid+'/tree')
            before=time.perf_counter();pred['selected_nonlinear']=predict(test,tmodel);latencies['selected_nonlinear_seconds']=time.perf_counter()-before
            w=blend['config']['tree_weight'];pred['selected_ensemble']=(1-w)*pred['selected_linear']+w*pred['selected_nonlinear']
            for anchor,names,strength in [('refit_base16',RECIPES['base16'],1),('refit_half',RECIPES['base16'],.5),('refit_box',RECIPES['base_box'],1)]:
                if anchor=='refit_half':model=base_model
                else:model=self.fitted(train,names,{'family':'linear','l2':1.0},oid+'/'+anchor)
                if anchor=='refit_base16':base_model=model
                pred[anchor]=predict(test,model,strength)
            model_receipt={'selected_linear':serialize_model(selected_model) if selected_model else config,'base16':serialize_model(base_model),'tree':serialize_model(tmodel),'train_races':len(starts(train)[0]),'test_races':len(starts(test)[0]),'prediction_latency':latencies}
            write(self.out/f'{oid}_models.json',model_receipt)
            # Tree bytes are kept isolated as a local research artifact.
            import pickle
            with (self.out/f'{oid}_tree.pkl').open('xb') as f:pickle.dump(tmodel,f,protocol=5)
            mask_fav=rule_mask(test,pred['selected_linear'],rules['favorite']['rule']) if rules['favorite'] else np.zeros(len(test),bool)
            mask_out=rule_mask(test,pred['selected_linear'],rules['outsider']['rule']) if rules['outsider'] else np.zeros(len(test),bool)
            for i,r in enumerate(test):
                all_outer.append({**r,'outer':oid,'predictions':{k:float(v[i]) for k,v in pred.items()},'favorite_selected':bool(mask_fav[i]),'outsider_selected':bool(mask_out[i])})
            summaries.append({'outer':oid,'train_races':len(starts(train)[0]),'test_races':len(starts(test)[0]),'chosen_recipe':chosen['name'],'config':chosen['config'],'tree':tree['name'],'tree_weight':w})
            print('selection frozen and predictions sealed',oid,flush=True)
        write(self.out/'frozen_selections.json',selections)
        with (self.out/'outer_predictions.jsonl').open('x') as f:
            for row in all_outer:f.write(canonical(row)+'\n')
        self.ledger.append('ALL_OUTER_PREDICTIONS_SEALED',prediction_sha256=sha(self.out/'outer_predictions.jsonl'),outer_labels_used_for_choices=False)
        # First aggregate outer performance occurs only after all choices have frozen.
        from scripts.offline_systematic_summary import summarize
        self.stage('diagnostics',60)
        report=summarize(all_outer,self.out,self.ledger)
        write(self.out/'execution.json',{'fits':self.fit_count,'validation_trials':self.trial_count,'cpu_seconds':time.process_time()-self.cpu_started,'wall_seconds':time.monotonic()-self.started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'folds':summaries,'python':sys.version,'executable':sys.executable,'packages':{x:importlib.metadata.version(x) for x in ['numpy','scipy','scikit-learn']},'thread_env':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}})
        self.ledger.append('PROGRAMME_COMPLETE',fits=self.fit_count,validation_trials=self.trial_count,result_sha256=sha(self.out/'summary.json'),decision='finite stages completed; no post-outer expansion')
        print(json.dumps(report['overall'],indent=2))


def rule_mask(rows,p,rule):
    if rule is None:return np.zeros(len(rows),bool)
    market=np.array([r['market'] for r in rows]);delta=p-market;condition=rule['condition']
    if rule['kind']=='favorite':
        mask=np.zeros(len(rows),bool)
        for a,n in zip(*starts(rows)):
            field=market[a:a+n];ix=np.flatnonzero(field==field.max())
            if len(ix)==1:mask[a+ix[0]]=True
        mask &= delta<=-rule['threshold']
    else:mask=(np.array([r['odds'] for r in rows])>=rule['minimum_odds'])&(delta>=rule['threshold'])
    if condition=='limited':mask &= np.array([r['features']['limited_history']>0 for r in rows])
    if condition=='layoff21':mask &= np.array([(r['features']['layoff_21'] or 0)>0 for r in rows])
    if condition=='pace_pressure':mask &= np.array([(r['features']['faster_neighbors'] or 0)>=1 for r in rows])
    if condition=='improving_margin':mask &= np.array([(r['features']['margin_recent_long_gap'] or 0)<0 for r in rows])
    if condition=='fast_time':mask &= np.array([r['features']['time_field_rank'] is not None and r['features']['time_field_rank']<=.5 for r in rows])
    return mask


def select_rules(rows,p,ledger,outer):
    market=np.array([r['market'] for r in rows]);y=np.array([r['y'] for r in rows]);trials=[]
    for t in [0,.01,.02,.05]:
        for c in ['any','limited','layoff21','pace_pressure','improving_margin']:
            trials.append({'kind':'favorite','threshold':t,'condition':c})
    for odds in [5,8,10,15]:
        for t in [0,.005,.01,.02]:
            for c in ['any','improving_margin','fast_time']:
                trials.append({'kind':'outsider','minimum_odds':odds,'threshold':t,'condition':c})
    eligible=collections.defaultdict(list)
    for rule in trials:
        mask=rule_mask(rows,p,rule);n=int(mask.sum());dates=len({r['race_date'] for r,m in zip(rows,mask) if m});nr=len({r['race_id'] for r,m in zip(rows,mask) if m})
        gain=float(np.mean((y[mask]-market[mask])**2-(y[mask]-p[mask])**2)) if n else None
        qualified=n>=20 and nr>=20 and dates>=5 and gain is not None and gain>0
        record={'rule':rule,'selections':n,'selected_races':nr,'dates':dates,'mean_binary_brier_improvement':gain,'qualified':qualified}
        ledger.append('SELECTION_RULE_VALIDATION',outer=outer,**record)
        if qualified:eligible[rule['kind']].append(record)
    return {kind:max(eligible[kind],key=lambda t:(t['mean_binary_brier_improvement'],t['selections'])) if eligible[kind] else None for kind in ['favorite','outsider']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--features',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    audit=json.loads((args.features/'feature_audit.json').read_text());datafile=args.features/'features.jsonl'
    if sha(datafile)!=audit['feature_hash']:raise ValueError('feature hash drift')
    for line in datafile.open():
        if scalar(line,'race_date')>'2026-07-09':raise ValueError('unauthorised date before label decode')
    data=[json.loads(line) for line in datafile.open()]
    args.out.mkdir(exist_ok=False);(args.out/'inner_predictions').mkdir()
    protocol={'outer':OUTER,'recipes':RECIPES,'trees':TREE_CONFIGS,'staged_linear_tuning':{'top_groups':3,'l2':[.1,.3,1,3],'strength':[.25,.5,1]},'market_powers':[.75,1,1.25,1.5],'ensemble_tree_weights':[0,.25,.5,1],'selection_rules':{'favorite_thresholds':[0,.01,.02,.05],'favorite_conditions':['any','limited','layoff21','pace_pressure','improving_margin'],'outsider_odds':[5,8,10,15],'outsider_thresholds':[0,.005,.01,.02],'outsider_conditions':['any','improving_margin','fast_time'],'qualification':'>=20 runners AND >=20 races AND >=5 dates AND positive binary-Brier gain on inner OOF; no qualified rule=>abstain'},'confirmation':'NONE: all outer evaluation dates previously inspected development','population_sha256':sha(datafile),'runner_sha256':sha(Path(__file__)),'source_base':'e09c16e78bd5824b4494602204b9af44b4cc4e94','stopping':'no new recipes or thresholds after outer metrics; shortlist at most3 selector families'}
    write(args.out/'SEARCH_PROTOCOL.json',protocol)
    search=Search(args.out,data)
    try:search.run()
    except Exception as exc:
        search.ledger.append('PROGRAMME_FAILED',error=str(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
