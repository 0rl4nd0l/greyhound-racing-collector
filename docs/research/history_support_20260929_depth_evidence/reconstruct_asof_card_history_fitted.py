"""One fixed development history-depth reconstruction from earlier admitted cards.

No current DB, operational bundles, protected targets, or post-cutoff cards.
A same-day disagreement excludes the entire target field. Identity is existing
canonical dog token, not claimed provider ID; uncertain same-day starts excluded.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
from scripts import offline_form_packet as form
from scripts.audit_development_form_quality import admitted_sources
from scripts.explain_market_residual import load_scope, gate_lines, FOUNDATION, sha, write


def union_history(current, prior):
    by_date=defaultdict(dict)
    for h in current+prior:
        key=tuple((k,str(v)) for k,v in sorted(h.items()))
        by_date[h['date']][key]=h
    if any(len(v)>1 for v in by_date.values()):
        return None
    return sorted((next(iter(v.values())) for v in by_date.values()),key=lambda h:h['date'],reverse=True)[:20]


def eligible_prior_history(pool, capture, target_date):
    admitted=[(t,rid,h) for t,rid,h in pool if t<capture]
    prior=[h for _,_,hh in admitted for h in hh if h['date']<target_date]
    return admitted,prior


def features(row, history, venue, distance, grade, size):
    raw=form.canonical.feature_row(row['race_id'],date.fromisoformat(row['race_date']),venue,distance,grade,size,row['box'],row['dog_token'],history)
    f={name:form._number(raw.get(form.ALIASES.get(name,name))) for name in form.FEATURES}
    finishes=[h['finish'] for h in history[:5] if h['finish'] is not None]
    f['recent_finish_best_5']=float(min(finishes)) if finishes else None
    return f


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    allowed,pins=load_scope()
    payload=(FOUNDATION/'development.jsonl').read_bytes()
    if sha(payload)!=pins[str(FOUNDATION/'development.jsonl')]:raise ValueError('development pin changed')
    rows=gate_lines(payload,allowed)
    pinfile=Path(__file__).resolve().parents[1]/'docs/research/market_explanation_20260929_inputs.json'
    expected=json.loads(pinfile.read_bytes()); path=FOUNDATION/'form_provenance.json'
    payload=form._verified(path,expected[str(path)])
    sources=admitted_sources(json.loads(payload),allowed)
    grouped=defaultdict(list)
    for r in rows:grouped[r['race_id']].append(r)
    cards=[]
    for rid,rr in sorted(grouped.items()):
        source=sources[rid]
        card=form._verified(Path(source['card_source_path']),source['card_source_sha256'],int(source['card_source_bytes']))
        sidecar=form._verified(Path(source['card_sidecar_path']),source['card_sidecar_sha256'],int(source['card_sidecar_bytes']))
        meta=json.loads(sidecar)
        capture=form.canonical.capture_timestamp(meta,require_timezone=True)
        jump=form.canonical.sidecar_jump_timestamp(meta,rid)
        form._validate_source_timing(meta,rid,capture,jump)
        if meta['runner_completeness']['status']!='COMPLETE':raise ValueError('incomplete current field')
        if (jump-capture).total_seconds()<3600 or meta.get('metadata_is_leakage_safe') is not True or meta['content_sha256']!=sha(card):raise ValueError('timing or identity')
        roster=sorted((r['box'],r['dog_token']) for r in rr)
        if sorted(form.canonical.parse_card_target_roster_bytes(card,source=rid))!=roster or sorted(form.canonical.sidecar_roster(meta,source=rid))!=roster:raise ValueError('roster')
        blocks=form.canonical.parse_form_blocks_bytes(card,source=rid)
        v,_,g,_=form.canonical.target_metadata({'metadata':meta},rid)
        d=form._metres(meta.get('target_distance') or meta.get('race_info',{}).get('distance'))
        histories={}
        for r in rr:
            h,rejected=form.canonical.accepted_history(blocks[r['dog_token']],date.fromisoformat(r['race_date']))
            if any(x['date']>capture.date() for x in h):raise ValueError('history after availability')
            if features(r,h,v,d,g,len(rr))!={f:r['features'][f] for f in form.FEATURES}:raise ValueError('baseline replay')
            histories[r['dog_token']]=h
        cards.append((capture,rid,rr,histories,(v,d,g),source['card_source_sha256'],source['card_sidecar_sha256']))
    cards.sort(key=lambda x:(x[0],x[1]))
    pools=defaultdict(list); pairs=[]; exclusions=[]; counts=Counter(); changes=Counter(); membership=[]
    for capture,rid,rr,histories,context,pin,sidecar_pin in cards:
        racepairs=[]; conflict=False
        for r in rr:
            token=r['dog_token']; current=histories[token]
            admitted,prior=eligible_prior_history(pools[token],capture,date.fromisoformat(r['race_date']))
            union=union_history(current,prior)
            if union is None:conflict=True;continue
            merged=features(r,union,*context,len(rr))
            changed=[f for f in form.FEATURES if merged[f]!=r['features'][f]]
            racepairs.append({'race_id':rid,'box':r['box'],'short_features':{f:r['features'][f] for f in form.FEATURES},'richer_features':merged,'short_count':len(current),'richer_count':len(union),'changed_features':changed,'prior_card_races':sorted({oldrid for _,oldrid,_ in admitted}),'card_capture':capture.isoformat(),'history_max_date':max(h['date'] for h in union).isoformat() if union else None})
        if conflict:exclusions.append({'race_id':rid,'reason':'CONFLICTING_OR_AMBIGUOUS_SAME_DAY_HISTORY'})
        else:
            pairs.extend(racepairs);counts['eligible_races']+=1;counts['eligible_runners']+=len(rr)
            counts['enriched_races']+=int(any(r['richer_count']>r['short_count'] for r in racepairs))
            counts['enriched_runners']+=sum(r['richer_count']>r['short_count'] for r in racepairs)
            counts['added_starts']+=sum(r['richer_count']-r['short_count'] for r in racepairs)
            for r in racepairs:changes.update(r['changed_features'])
        for token,h in histories.items():pools[token].append((capture,rid,h))
        membership.append({'race_id':rid,'capture':capture.isoformat(),'source_sha256':pin,'sidecar_sha256':sidecar_pin})
    counts['original_races']=len(cards);counts['original_runners']=len(rows);counts['excluded_races']=len(exclusions)
    write(out/'summary.json',{'counts':dict(counts),'feature_changes':dict(changes),'exclusions':exclusions,'new_fits':0,'interpretation':'Earlier retained-card history reconstruction, not production DB replay','policy':'strictly earlier capture, no same-day disagreement, unchanged canonical formulas, cap20','identity_limit':'canonical dog token; no independent stable provider runner identifier'})
    write(out/'membership.json',membership)
    with (out/'paired_features.jsonl').open('x') as f:
        for r in pairs:f.write(json.dumps(r,sort_keys=True)+'\n')
    print(json.dumps(dict(counts)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
