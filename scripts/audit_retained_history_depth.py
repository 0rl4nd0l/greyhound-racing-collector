"""Exact retained operational input diagnostics. Never reads target outcomes.

History values remain machine-only. Output contains support counts and counts of
feature differences, never raw histories, their labels, or per-runner rates.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'docs/research/history_support_20260929_history_inputs.json'
FEATURES = ['prior_start_count','days_since_last_start','recent_finish_mean_3',
'recent_finish_best_5','recent_win_rate_5','recent_place_rate_5','recent_avg_margin_5',
'career_win_rate','career_place_rate','career_avg_finish','starts_same_venue',
'win_rate_same_venue','starts_same_distance','win_rate_same_distance',
'same_grade_start_count','same_grade_win_rate']


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def verified(path, expected):
    payload = Path(path).read_bytes()
    if digest(payload) != expected:
        raise ValueError('input identity changed')
    return payload


def safe_path(root, relative):
    p = (root / relative).resolve()
    if root.resolve() not in p.parents:
        raise ValueError('path outside retained bundle')
    return p


def known_context(rows, venue, distance, grade):
    """Counts use production definitions; exact distance separately identified."""
    from scripts.run_feature_recovery_execution_v1 import normalize_grade
    subsets = {'all':list(rows), 'venue':[r for r in rows if str(r.get('venue') or '').upper() == venue.upper()],
        'distance_exact':[r for r in rows if r.get('distance_num') == distance],
        'distance_tolerance':[r for r in rows if r.get('distance_num') is not None and distance is not None and abs(r['distance_num']-distance)<=50],
        'grade':[r for r in rows if grade and normalize_grade(r.get('grade_normalized') or r.get('grade')) == normalize_grade(grade)]}
    return {k:{'starts':len(v), 'known_finishes':sum(r.get('finish_num') is not None for r in v),
               'runners_no_matching_starts':int(not v),
               'runners_no_known_finish':int(not any(r.get('finish_num') is not None for r in v))}
            for k,v in subsets.items()}


def merge_diagnostics(db, card, merge):
    merged = merge(db,card)
    coarse = Counter((r.get('race_date'), str(r.get('venue') or '').upper(), r.get('distance_num')) for r in merged)
    return merged, {'db_rows':len(db),'card_rows':len(card),'merged_rows':len(merged),
        'duplicate_rows_removed':len(db)+len(card)-len(merged),
        'retained_db_rows':sum(any(r is d for d in db) for r in merged),
        'possible_same_start_conflict_groups':sum(v>1 for v in coarse.values())}


def compare_bundle(root, manifest, completion_hash):
    # These imports have no data reads. Both source files are verified against
    # the exact retained production archive before any history is decoded.
    from scripts import run_shadow_non_tgr_rf_evaluation as live
    from scripts import run_feature_recovery_execution_v1 as feature
    from scripts import build_form_only_v1_packet as canonical
    from scripts.offline_form_packet import ALIASES, _metres
    files=manifest['files']
    paths={role:safe_path(root,v['path']) for role,v in files.items()}
    for role in ('normalized_form','form_metadata','generator_source_archive','feature_schema'):
        verified(paths[role],files[role]['sha256'])
    with zipfile.ZipFile(paths['generator_source_archive']) as archive:
        for module in (live,feature):
            relative='scripts/'+Path(module.__file__).name
            if digest(archive.read(relative)) != digest(Path(module.__file__).read_bytes()):
                raise ValueError('production source mismatch')
    history=safe_path(root,manifest['history']['path'])
    verified(history,manifest['history']['sha256'])
    seal=json.loads(verified(safe_path(root,manifest['history_seal']['path']),manifest['history_seal']['sha256']))
    if seal['target_rows_materialized'] or seal['at_or_after_cutoff_rows_materialized'] or seal['target_race_id'] != manifest['race_id'] or seal['sealed_sha256'] != manifest['history']['sha256']:
        raise ValueError('history seal scope failed')
    completion=json.loads(verified(root/'completion.json',completion_hash))
    if completion['manifest_sha256']!=digest((root/'manifest.json').read_bytes()):raise ValueError('seal manifest identity')
    if not datetime.fromisoformat(completion['inputs_sealed_at']) < datetime.fromisoformat(manifest['prediction_cutoff']):raise ValueError('durable seal after cutoff')
    jump=datetime.fromisoformat(manifest['jump_at'])
    if not datetime.fromisoformat(manifest['capture_completed_at']) < datetime.fromisoformat(manifest['prediction_cutoff']) < jump:
        raise ValueError('input availability cutoff failed')
    target_date=jump.date().isoformat()
    formrows=live.load_live_csv(paths['normalized_form'])
    roster=[]
    for raw in formrows:
        name,box=live.parse_live_runner_identity(live.csv_value(raw,'dog_name','Dog Name','dog','runner','runner_name'),live.csv_value(raw,'box_number','box','Box'))
        if name:roster.append((name,box))
    keys={live.clean_name(name) for name,_ in roster}
    # Identity-only SQL first: the complete sealed DB is already active-runner
    # scoped, but verify that before calling the original machine feature loader.
    con=sqlite3.connect(f'file:{history}?mode=ro',uri=True)
    con.row_factory=sqlite3.Row
    try:
        names={live.clean_name(r[0]) for r in con.execute('SELECT DISTINCT dog_name FROM dog_race_data')}
        if not names <= keys:raise ValueError('unexpected runner in retained database')
        db=feature.load_db_history(con)
    finally:con.close()
    if any(r['race_date']>=target_date for rr in db.values() for r in rr):raise ValueError('invalid sealed cutoff')
    card=live.live_form_history_by_dog(formrows,target_race_date=target_date)
    metadata=json.loads(paths['form_metadata'].read_bytes())
    context=live.load_live_sidecar_context(paths['normalized_form'])
    venue=str(context['race_info']['venue'])
    distance=feature.safe_float(context.get('target_distance'))
    grade=feature.normalize_grade(context.get('target_grade'))
    if not venue or distance is None or not grade:raise ValueError('target context unavailable')
    target={'distance':distance,'grade':grade}
    schema=json.loads(paths['feature_schema'].read_bytes())
    actual=live.build_live_feature_rows(input_paths=[paths['normalized_form']],schema=schema,db_path=history)
    projected=[{'race_id':r['race_id'],'dog_name':r['dog_name'],'box_number':r['box_number'],'features':{name:r[name] for name in FEATURES}} for r in actual]
    projected.sort(key=lambda r:(r['race_id'],r['box_number'],r['dog_name']))
    saved=json.loads(verified(root/'feature_values.json',manifest['feature_values_sha256']))
    if projected != saved:raise ValueError('retained feature replay mismatch')
    totals=Counter(); depth=Counter(); formulas=Counter(); length=Counter(); contexts={'card':{},'merged':{}}; reason=Counter()
    blocks=canonical.parse_form_blocks_bytes(paths['normalized_form'].read_bytes(),source=manifest['race_id'])
    cv,cd,cg,_=canonical.target_metadata({'metadata':metadata},manifest['race_id'])
    cd=_metres(metadata.get('target_distance') or metadata.get('race_info',{}).get('distance'))
    for name,box in roster:
        key=live.clean_name(name); cr=card.get(key,[]); dr=db.get(key,[])
        merged,diag=merge_diagnostics(dr,cr,live.merge_prior_history_rows)
        totals.update(diag); totals['runners']+=1
        totals['runners_with_added_rows']+=int(len(merged)>len(cr)); totals['runners_with_db_rows']+=int(bool(dr))
        length[f'{len(cr)}->{len(merged)}']+=1
        for route,rr in [('card',cr),('merged',merged)]:
            for subset,counts in known_context(rr,venue,distance,grade).items():
                contexts[route].setdefault(subset,Counter()).update(counts)
        cf={};mf={}
        feature.add_history_features(cf,live.merge_prior_history_rows([],cr),target,target_date,venue)
        feature.add_history_features(mf,merged,target,target_date,venue)
        for f in FEATURES:depth[f]+=int(cf.get(f)!=mf.get(f))
        token=canonical.dog_token(name)
        ch,rejected=canonical.accepted_history(blocks[token],date.fromisoformat(target_date))
        reason.update(why for why,_ in rejected)
        totals['canonical_card_accepted_rows']+=len(ch)
        canonical_row=canonical.feature_row(manifest['race_id'],date.fromisoformat(target_date),cv,cd,cg,len(roster),box,token,ch)
        canonical_values={f:feature.safe_float(canonical_row.get(ALIASES.get(f,f))) for f in FEATURES}
        finishes=[h['finish'] for h in ch[:5] if h['finish'] is not None]
        canonical_values['recent_finish_best_5']=min(finishes) if finishes else None
        for f in FEATURES:
            a,b=canonical_values.get(f),cf.get(f)
            formulas[f]+=int((a is None)!=(b is None) or (a is not None and b is not None and not math.isclose(a,b,abs_tol=1e-7,rel_tol=0)))
        totals['runners_feature_changed_from_db']+=int(any(cf.get(f)!=mf.get(f) for f in FEATURES))
        totals['card_unknown_finishes']+=sum(r.get('finish_num') is None for r in cr)
        totals['merged_unknown_finishes']+=sum(r.get('finish_num') is None for r in merged)
        totals['research_cap_would_remove_rows']+=max(0,len(merged)-20)
    return {'race_id':manifest['race_id'],'matched_runners':len(roster),'counts':dict(totals),
        'history_count_pairs':dict(length),'contexts':contexts,'same_formula_db_feature_changes':dict(depth),'same_card_route_definition_changes':dict(formulas),'canonical_rejections':dict(reason),
        'retained_feature_replay':'EXACT','availability':'retained_before_prediction_cutoff','inputs_sealed_at':completion['inputs_sealed_at'],'prediction_cutoff':manifest['prediction_cutoff'],
        'manifest_sha256':digest((root/'manifest.json').read_bytes()),'history_sha256':manifest['history']['sha256']}


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    spec=json.loads(SPEC.read_bytes())
    for kind in ('report','authority'):verified(spec[kind]['path'],spec[kind]['sha256'])
    authority=json.loads(Path(spec['authority']['path']).read_bytes())
    if authority['result_access'] is not False or authority['history'] != 'machine-only strictly earlier histories under existing isolation; protected histories allowed only as feature inputs':raise ValueError('unexpected authority')
    if len(spec['manifests']) != 3:raise ValueError('unexpected population')
    # Load reservation metadata before histories (this helper does not decode
    # labels). Operational targets are separate, never development labels.
    from scripts.explain_market_residual import load_scope
    allowed,reservation_pins=load_scope()
    manifests=[(Path(p['path']),json.loads(verified(p['path'],p['sha256']))) for p in spec['manifests']]
    expected={'Race 2 - BULLI - 2026-09-29','Race 7 - GEE - 2026-09-29','Race 6 - HOR - 2026-09-29'}
    if {m['race_id'] for _,m in manifests} != expected:raise ValueError('operational identity scope changed')
    def offline(event,params):
        if event.startswith('socket.') or event in {'subprocess.Popen','os.system'}:raise PermissionError('offline only')
        if event=='sqlite3.connect' and str(params[0]) not in {f'file:{p.parent / m["history"]["path"]}?mode=ro' for p,m in manifests}:raise PermissionError('unretained DB')
    sys.addaudithook(offline)
    result={'schema':'retained_history_depth_diagnostic_v1','target_results_accessed':0,'new_fits':0,
        'development_races':len({r for r,_ in allowed}),'reservation_input_hashes':reservation_pins,
        'operational_bundles':[compare_bundle(p.parent,m,item['completion_sha256']) for (p,m),item in zip(manifests,spec['manifests'])]}
    with args.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'completed_bundles':3,'output':str(args.output)}))

if __name__=='__main__':
    try:run()
    except Exception as exc:
        # Do not expose source exceptions/payloads from protected input parsing.
        print(json.dumps({'status':'FAILED','exception_type':type(exc).__name__}),file=sys.stderr)
        raise SystemExit(2)
