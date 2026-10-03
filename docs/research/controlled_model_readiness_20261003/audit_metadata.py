"""Bounded metadata projection only; no scoring, DB opens or provider access.

The parser derives from research_input_firewall.project_metadata, extending its
array wildcard to literal object keys for file-hash manifests. Unknown scalar
values are skipped without decoding. Immutable role/field limits live in manifest.
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
class InputBoundary(ValueError): pass

def project_metadata(raw, paths):
    """Decode only explicitly named scalar paths; unknown payloads stay bytes.

    Paths use * for array positions. No scalar string is recursively decoded.
    Duplicate keys reject, including in opaque subtrees. This is a projection,
    not a claim that the original document contains no results.
    """
    allowed={tuple(path.split(".")) for path in paths}; i=0; absent=object()
    def matches(path):
        return any(len(rule)==len(path) and all(a=='*' or a==b for a,b in zip(rule,path)) for rule in allowed)
    def space():
        nonlocal i
        while i<len(raw) and raw[i] in b" \t\r\n":i+=1
    def token_string():
        nonlocal i
        start=i;i+=1
        while i<len(raw):
            if raw[i]==92:i+=2
            elif raw[i]==34:i+=1;return raw[start:i]
            else:i+=1
        raise InputBoundary('JSON_FRAMING')
    def parse(path,depth=0):
        nonlocal i
        if depth>50:raise InputBoundary('JSON_DEPTH')
        space();kind=raw[i:i+1]
        if kind in (b'{',b'['):
            if matches(path) and not any(p[:len(path)]==path and len(p)>len(path) for p in allowed):raise InputBoundary('EXPECTED_METADATA_SCALAR:'+'.'.join(path))
            is_object=kind==b'{';closing=b'}' if is_object else b']';i+=1
            output={} if is_object else [];seen=set();space()
            if raw[i:i+1]==closing:i+=1;return output
            while True:
                space()
                if is_object:
                    if raw[i:i+1]!=b'"':raise InputBoundary('JSON_FRAMING')
                    key=json.loads(token_string())
                    if key in seen:raise InputBoundary('DUPLICATE_METADATA_KEY')
                    seen.add(key);space()
                    if raw[i:i+1]!=b':':raise InputBoundary('JSON_FRAMING')
                    i+=1
                else:key='*'
                child=parse(path+(key,),depth+1)
                if child is not absent and (not isinstance(child,(dict,list)) or child):
                    if is_object:output[key]=child
                    else:output.append(child)
                space();separator=raw[i:i+1];i+=1
                if separator==closing:return output
                if separator!=b',':raise InputBoundary('JSON_FRAMING')
        start=i
        if kind==b'"':token_string()
        else:
            while i<len(raw) and raw[i] not in b',]} \t\r\n':i+=1
            if start==i:raise InputBoundary('JSON_FRAMING')
        return json.loads(raw[start:i]) if matches(path) else absent
    result=parse(());space()
    if i!=len(raw) or not isinstance(result,dict):raise InputBoundary('JSON_FRAMING')
    return result

def run_projection():
    root=Path(__file__).parent
    manifest=json.loads((root/'manifest.json').read_bytes())
    assert datetime.now(timezone.utc)<datetime.fromisoformat(manifest['deadline'])
    assert manifest['actual_file_count']<=manifest['finite_limits']['files']
    # Unknown values cannot be decoded: guarded synthetic check before reading.
    original=json.loads
    def guarded(raw,*args,**kwargs):
        assert 'DO_NOT_DECODE' not in str(raw)
        return original(raw,*args,**kwargs)
    json.loads=guarded
    try:
        assert project_metadata(b'{"safe":1,"secret":"DO_NOT_DECODE","predictions":[{"probability":"DO_NOT_DECODE","box":2}]}', ['safe','predictions.*.box'])=={'safe':1,'predictions':[{'box':2}]}
    finally:json.loads=original
    projected={};total=0
    for row in manifest['files']:
        path=Path(row['path'])
        assert path.resolve()==path and not path.is_symlink() and path.is_file()
        assert path.stat().st_size==row['bytes']<=manifest['finite_limits']['bytes_per_file']
        raw=path.read_bytes();total+=len(raw)
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        assert total<=manifest['finite_limits']['aggregate_bytes_per_pass']
        allow=manifest['projection_rules']['allowlisted_fields_by_role'][row['role']]
        if allow:projected[row['path']]=project_metadata(raw,allow)
    assert total==manifest['actual_bytes_per_pass']
    return manifest,projected

def audit(manifest, projected):
    import re
    def stamp(s):
        d=datetime.fromisoformat(s)
        assert d.utcoffset() is not None
        return d
    def canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
    def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
    def token(s):return re.sub('[^A-Z0-9]','',s.upper())
    inventory={row['path']:row for row in manifest['files']}
    original=next(r for r in manifest['files'] if r['role']=='original_source_identity')
    origin_root=Path(original['path']).parent
    origin=projected[original['path']]
    origin_plan=projected[next(r['path'] for r in manifest['files'] if r['role']=='original_plan')]
    repo=Path(__file__).resolve().parents[3]
    bundle_root=Path(manifest['historical_root'])/'predictions/bundles'
    rows=[]; model_hashes=set(); model_manifest_hashes=set(); current_mismatch=set()
    for admission_entry in [r for r in manifest['files'] if r['role']=='admission']:
        path=Path(admission_entry['path']);a=projected[str(path)];c=projected[str(path.with_name('completion.json'))]
        bundle=bundle_root/c['bundle_entry']['directory']
        assert bundle.parent==bundle_root
        def get(relative):return projected[str(bundle/relative)]
        def ref(relative):return inventory[str(bundle/relative)]
        bm=get('bundle_manifest.json');files=bm['files'];q=get('request.json');p=get('result.json')
        comp=get('comparison/production.json');impl=get('features/sealed/implementation_file_manifest.json')
        feature=get('features/sealed/shadow_manifest.json');history=get('features/history_seal.json')
        mm=get('model/manifest.json');mh=get('model/model.json');registry=get('comparison/registry.json')
        checks={}
        checks['completion_bound_to_admission']=all(a[k]==c[k] for k in ['prediction_id','job_id','race','runner_set_sha256','plan_sha256','retained_input_manifest_sha256']) and a['bundle_directory']==bundle.name
        checks['bundle_identity_and_digest']=bm['prediction_id']==a['prediction_id'] and bm['job_id']==a['job_id'] and c['bundle_entry']['manifest_sha256']==ref('bundle_manifest.json')['sha256'] and c['bundle_entry']['logical_bundle_sha256']==digest(bm)
        checks['all_inventoried_bundle_roles_hash_bound']=all(files[str(Path(r['path']).relative_to(bundle))]=={'bytes':r['bytes'],'sha256':r['sha256']} for r in manifest['files'] if Path(r['path']).is_relative_to(bundle) and r['role']!='bundle_manifest')
        required=['features/sealed_history.db','features/sealed/shadow_feature_rows.json','source/capture.json','retained_inputs.zip','config.json','odds_receipt.json']
        forms=[name for name in files if name.startswith('source/') and name.endswith('.csv')]
        checks['required_payload_roles_declared']=all(name in files for name in required) and len(forms)==1 and forms[0]+'.metadata.json' in files
        checks['required_payload_files_present_with_declared_sizes']=all((bundle/name).is_file() and not (bundle/name).is_symlink() and (bundle/name).stat().st_size==files[name]['bytes'] for name in required+forms+[forms[0]+'.metadata.json'])
        checks['prejump_admission_and_publication']=stamp(a['admitted_at'])<=stamp(p['generated_at'])<=stamp(c['published_complete_at'])<stamp(a['decision_at']) and stamp(a['decision_at'])==stamp(c['decision_at']) and (stamp(a['race']['jump_timestamp'])-stamp(a['decision_at'])).total_seconds()==120 and c['status']=='COMPLETE_BEFORE_CUTOFF'
        checks['original_four_statuses_sealed']=c['models']==dict.fromkeys(['market','production','residual_box','residual_half'],'SEALED')
        checks['production_header_status']=p['status']=='PREDICTION_READY' and comp['status']=='SEALED' and comp['failure'] is None and comp['candidate']=='production'
        keys=sorted(f"{int(r['box_number'])}:{r['identity'].strip().upper()}" for r in q['runners'])
        checks['sealed_roster_hash_crosslinks']=len(keys)>=2 and len(keys)==len(set(keys)) and q['runner_set_sha256']==a['runner_set_sha256']==comp['runner_set_sha256']==p['evidence']['runner_set_sha256']
        # sealed_runner_set_sha256 binds full race+runners, including native IDs.
        # The immutable allowlist omitted native IDs: no reconstructed-hash claim.
        qr=sorted((r['box_number'],r['identity'],token(r['display_name'])) for r in q['runners'])
        cr=sorted((r['box_number'],r['identity'],token(r['dog_name'])) for r in comp['predictions'])
        pr=sorted((r['box_number'],token(r['dog_name'])) for r in p['prediction']['predictions'])
        checks['roster_metadata_equality']=qr==cr and [(b,n) for b,i,n in qr]==pr and all(1<=b<=10 for b,i,n in qr)
        checks['retained_manifest_identity']=q['retained_input_manifest_sha256']==a['retained_input_manifest_sha256']==comp['retained_input_manifest_sha256'] and bool(a['retained_input_manifest_sha256'])
        checks['comparison_admission_hash']=comp['admission_sha256']==admission_entry['sha256']
        ci=comp['input_identity'];lead=(stamp(a['race']['jump_timestamp'])-stamp(ci['captured_at'])).total_seconds()
        checks['exact_quote_time_metadata']=120<=lead<=600 and stamp(ci['captured_at'])<=stamp(a['admitted_at']) and get('protocol/collector_exact_receipt.json')['captured_at']==ci['captured_at']
        checks['capture_form_feature_hash_crosslinks']=ci['form_sha256']==files[forms[0]]['sha256'] and ci['sidecar_sha256']==files[forms[0]+'.metadata.json']['sha256'] and ci['capture_sha256']==files['source/capture.json']['sha256'] and ci['production_feature_rows_sha256']==files['features/sealed/shadow_feature_rows.json']['sha256'] and ci['odds_receipt_sha256']==files['odds_receipt.json']['sha256']
        checks['feature_freeze_prejump']=stamp(feature['feature_freeze_timestamp'])<=stamp(comp['completed_at'])<=stamp(c['published_complete_at'])
        cut=p['evidence']['authenticated_cutoff']
        checks['history_seal_crosslinks']=cut['history_seal_sha256']==ref('features/history_seal.json')['sha256'] and cut['sealed_sha256']==history['sealed_sha256']==files['features/sealed_history.db']['sha256'] and cut['source_sha256']==history['source_sha256'] and cut['cutoff_timestamp']==history['cutoff_timestamp']==a['race']['jump_timestamp']
        checks['history_exclusion_metadata']=history['target_race_id']==a['race']['race_id'] and history['cutoff_basis']=='race_date_strictly_before_target_jump_date' and history['target_rows_materialized']==history['at_or_after_cutoff_rows_materialized']==0
        checks['implementation_artifact_hash_crosslinks']=all(files[str(Path(name).relative_to(bundle))]==v for name,v in impl['artifact_files'].items())
        checks['original_implementation_hashes']=set(impl['implementation_files'])==set(impl['implementation_file_hashes']) and all(inventory[str(origin_root/name)]['sha256']==sha==origin['files'][name] for name,sha in impl['implementation_file_hashes'].items())
        for name,sha in impl['implementation_file_hashes'].items():
            if inventory[str(repo/name)]['sha256']!=sha:current_mismatch.add(name)
        modelhash=ref('model/model.json')['sha256'];model_hashes.add(modelhash);model_manifest_hashes.add(ref('model/manifest.json')['sha256'])
        checks['production_model_bindings']=modelhash==mm['model_sha256']==comp['model_sha256']==q['model']['model_sha256']==p['model']['artifact_sha256']==registry['production']['artifacts/frozen_models/market_form_residual_v1/model.json'] and ref('model/manifest.json')['sha256']==q['model']['manifest_sha256']==p['model']['artifact_manifest_sha256']==registry['production']['artifacts/frozen_models/market_form_residual_v1/manifest.json'] and p['model']['resolved']=='market_form_residual_v1'
        checks['comparison_registry_hash']=get('comparison/plan.json')['candidate_registry']['sha256']==ref('comparison/registry.json')['sha256']
        checks['same_frozen_base_full_half_supported']=mm['derivation_contract']=={'full_strength':1.0,'half_strength':0.5,'shared_base_model_count':1,'variants_are_not_separate_models':True} and mh['algorithm']=={'residual_cap':.35,'strengths':{'full':1.0,'half':.5},'within_race_centering':True} and len(mh['feature_contract']['feature_order'])==16 and len(mh['feature_contract']['expanded_feature_order'])==32 and mh['fit']['population_sha256']==mm['fit_population_sha256']
        rows.append({'admission_key':path.parent.name,'prediction_id':a['prediction_id'],'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],'runner_count':len(keys),'prediction_lead_seconds':(stamp(a['race']['jump_timestamp'])-stamp(c['published_complete_at'])).total_seconds(),'quote_lead_seconds':lead,'admission_manifest_entry_sha256':admission_entry['sha256'],'bundle_manifest_sha256':ref('bundle_manifest.json')['sha256']})
    current=Path(manifest['current_day_separate_census']['root']);observed=datetime.now(timezone.utc).isoformat()
    admissions=sorted(current.glob('*/admission.json')) if current.is_dir() else []
    assert len(admissions)<=manifest['current_day_separate_census']['max_paths']
    return {'schema_version':'controlled_model_metadata_readiness_report_v1','observed_at':observed,'manifest_sha256':hashlib.sha256((Path(__file__).parent/'manifest.json').read_bytes()).hexdigest(),'status':'METADATA_BINDINGS_VERIFIED_WITH_EXPLICIT_UNKNOWNS' if all(not r['failed_checks'] for r in rows) else 'HOLD_METADATA_CHECK_FAILURE','historical_cohort_count':len(rows),'metadata_checks_passed_count':sum(not r['failed_checks'] for r in rows),'races':rows,'source_binding':{'original_commit':origin['commit'],'plan_commit':origin_plan['commit'],'original_source_identity_matches_plan':original['sha256']==origin_plan['source_identity_sha256'],'original_feature_implementation_verified':all(r['checks']['original_implementation_hashes'] for r in rows),'present_worktree_mismatches':sorted(current_mismatch),'present_implementation_substituted':False},'model_binding':{'model_sha256':sorted(model_hashes),'manifest_sha256':sorted(model_manifest_hashes),'one_shared_base_across_cohort':len(model_hashes)==len(model_manifest_hashes)==1,'frozen_production_full_half_supported':all(r['checks']['same_frozen_base_full_half_supported'] for r in rows),'new_full_half_predictions_computed':0,'new_full_half_predictions_sealed':0},'current_day_separate':{'root':str(current),'observed_at':observed,'root_exists':current.is_dir(),'admission_files_count':len(admissions),'admission_keys':[p.parent.name for p in admissions],'included_in_historical_cohort':False,'scope':'point_in_time_filename_census_only_no_future_membership_claim'},'unknown_or_not_reverified':['Native-ID-inclusive sealed roster hash reconstruction is UNKNOWN: the immutable allowlist omits source_native_runner_id; only declared hash equality and box/name/identity equality were verified','Opaque form, history DB, feature-row, capture and retained-archive payloads were not decoded or hashed; declared hashes, file presence and byte counts only were cross-checked','Official dog ID/reserve substitution against source/result evidence was not reverified; retained prediction roster equality is not that proof','Exact odds values and odds-receipt roster were not decoded; capture timing and receipt hashes only','No numerical score replay, coefficient validation, preprocessing-value validation, result closure or performance evaluation','No new production-half paired seal or research membership; prospective authorization and reservation clearance remain required'],'read_scope':{'metadata_source_read_passes':3,'manifest_files':manifest['actual_file_count'],'bytes_per_pass':manifest['actual_bytes_per_pass'],'requests':0,'fits':0,'scores':0,'result_bodies_decoded':0,'probabilities_decoded':0,'history_databases_opened':0,'form_rows_decoded':0,'activation':False}}


if __name__=='__main__':
    output=Path(__file__).with_name('readiness-report.json')
    if output.exists():raise SystemExit('Immutable report already exists; do not repeat the consumed audit')
    manifest,projected=run_projection()
    result=audit(manifest,projected)
    result['audit_interpretation_correction']={
        'initial_draft':'initial-hash-interpretation-failure.json',
        'initial_draft_preserved':True,
        'reason':'Initial audit used receipt runner_set_sha256 for the different sealed_runner_set_sha256 schema; source identity digest also used a partial projection instead of its frozen raw-byte hash',
        'correct_rule':'on_demand.sealed_runner_set_sha256 hashes complete race plus runner records; compare pinned hash references and allowlisted identities only, mark native-ID reconstruction UNKNOWN. Compare source identity raw hash from immutable inventory to plan pin.',
        'immutable_manifest_widened':False,'original_evidence_modified':False}
    result['audit_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    output.chmod(0o444)
    print(json.dumps({'status':result['status'],'cohort':result['historical_cohort_count'],
        'metadata_checks_passed':result['metadata_checks_passed_count'],
        'source_binding':result['source_binding'],'current_day_separate':result['current_day_separate'],
        'prediction_lead_seconds_range':[min(r['prediction_lead_seconds'] for r in result['races']),max(r['prediction_lead_seconds'] for r in result['races'])]}))
