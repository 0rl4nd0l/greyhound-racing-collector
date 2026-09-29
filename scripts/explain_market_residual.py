"""Offline, identity-gated explanation of retained #193 forecasts; no fitting."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/l4nd0/greyhound-offline-systematic-output-20260924')
FOUNDATION = SOURCE/'foundation'
PINS = {
    'development.jsonl': 'c58bd59bc0d52666d31812dccd60981de7f2f03b64862f98ddf9f88ef60cec46',
    'protected_records.json': '9fddce8fa70ea96c4d7a6c33ef61594ddeb47e254cd08bb87555b2d94fc4da88',
    'dataset_assessment.json': 'b0737c5b1fcf27ba34333ea824bbc363ab4d4a5c9986bc6b04b453ed81e16bea',
}
MODELS = ['refit_base16', 'refit_half', 'refit_box']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def scalar(line, field):
    matches = re.findall(r'"' + re.escape(field) + r'"\s*:\s*("(?:[^"\\]|\\.)*"|[0-9]+)', line)
    values = [json.loads(v) for v in matches]
    if not values or any(v != values[0] for v in values):
        raise ValueError('missing or ambiguous identity: ' + field)
    return values[0]


def race_key(rid):
    match = re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})', rid)
    if not match:
        raise ValueError('invalid race identity')
    return f'{match[3]}|{match[2]}|{int(match[1])}'


def load_scope():
    pins = {}
    def verified(path, expected):
        data = path.read_bytes()
        if sha(data) != expected:
            raise ValueError('source hash changed: ' + str(path))
        pins[str(path)] = expected
        return data
    deny = json.loads(verified(FOUNDATION/'protected_records.json', PINS['protected_records.json']))['records']
    incident = json.loads(verified(Path('/home/l4nd0/greyhound-offline-prediction-20260924/docs/research/offline_20260924_access_incident.json'),
        '724b4bf449aaa17f3f1419b579bf38ea4e836a776a6b1366339c44f821eb5f58'))
    assessment = json.loads(verified(FOUNDATION/'dataset_assessment.json', PINS['dataset_assessment.json']))
    original = set()
    for name, pin in assessment['inputs'].items():
        payload = verified(Path(name), pin)
        if name.endswith('pre-outcome-manifest-v2.json'):
            original.update(r['race_key'] for r in json.loads(payload)['candidates'])
        elif name.endswith('out_of_time_races.csv'):
            original.update(race_key(r['race_id']) for r in csv.DictReader(payload.decode().splitlines()))
        elif name.endswith('frozen_august_odds_cohort.jsonl'):
            original.update(race_key(scalar(line, 'race_id')) for line in payload.decode().splitlines())
        # Mixed canonical market files are byte-hashed, never decoded here.
    if original != set(deny) or min(k[:10] for k in deny) <= '2026-07-09':
        raise ValueError('reservation union changed or overlaps historical scope')
    if not all(race_key(rid) in deny for rid in incident['identities']):
        raise ValueError('incident identity not protected')
    identities = {}
    for line in verified(FOUNDATION/'development.jsonl', PINS['development.jsonl']).decode().splitlines():
        rid, day = scalar(line, 'race_id'), scalar(line, 'race_date')
        box = scalar(line, 'box')
        if race_key(rid) in deny or rid in incident['identities'] or not '2026-06-10' <= day <= '2026-07-09' or race_key(rid)[:10] != day:
            raise ValueError('ineligible identity before decoding')
        if (rid, box) in identities or not 1 <= box <= 8:
            raise ValueError('invalid or duplicate box')
        identities[rid, box] = scalar(line, 'dog_token')
    if len(identities) != 2360 or len({rid for rid, _ in identities}) != 331:
        raise ValueError('development membership changed')
    return identities, pins


def gate_lines(payload, allowed):
    lines = payload.decode().splitlines()
    seen = set()
    for line in lines:
        rid, box = scalar(line, 'race_id'), scalar(line, 'box')
        key = rid, box
        if key not in allowed or key in seen or scalar(line, 'dog_token') != allowed[key]:
            raise ValueError('unadmitted/duplicate runner before record decode')
        if scalar(line, 'race_date') != race_key(rid)[:10]:
            raise ValueError('identity date mismatch')
        seen.add(key)
    for rid in {key[0] for key in seen}:
        if {key for key in seen if key[0] == rid} != {key for key in allowed if key[0] == rid}:
            raise ValueError('incomplete admitted field')
    return [json.loads(line) for line in lines]


def losses(y, p):
    y, p = np.asarray(y), np.asarray(p, float)
    if (len(y) != len(p) or not np.all(np.isin(y, [0, 1])) or y.sum() != 1
            or np.any(~np.isfinite(p)) or np.any(p <= 0) or not np.isclose(p.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError('invalid complete-field probability/label')
    top = np.isclose(p, p.max(), atol=1e-12, rtol=0)
    return {'ll': float(-np.log(p[y == 1])[0]), 'brier': float(((p-y)**2).sum()),
            'accuracy': float(y[top].sum()/top.sum()), 'top_ties': int(top.sum())}


def decomposition(rows, model, strength=1):
    prep = model['prep']
    names = prep['names']
    raw = np.array([[r['features'].get(f, np.nan) for f in names] for r in rows], float)
    missing = ~np.isfinite(raw)
    expanded = np.c_[np.where(missing, prep['median'], raw), missing]
    x = (expanded-np.array(prep['mean']))/np.array(prep['scale'])
    if not prep['center']:
        raise ValueError('expected centered residual receipt')
    x -= x.mean(axis=0)
    contributions = x*np.array(model['beta'])
    z = contributions.sum(axis=1)
    cap = strength*.35*np.tanh(z/.35)
    market = np.array([r['market'] for r in rows])
    log_normalizer = float(np.log(np.sum(market*np.exp(cap))))
    p = market*np.exp(cap-log_normalizer)
    return {'names': names + ['missing::'+f for f in names], 'raw': raw, 'missing': missing,
            'x': x, 'contributions': contributions, 'z': z, 'cap': cap,
            'cap_effect': cap-strength*z, 'log_normalizer': log_normalizer, 'p': p}


def characteristics(rows, names, grade):
    m = np.array([r['market'] for r in rows])
    p = np.array([r['predictions']['refit_base16'] for r in rows])
    fav = np.flatnonzero(np.isclose(m, m.max(), atol=1e-12, rtol=0))
    entropy = float(-np.sum(m*np.log(m))/np.log(len(m)))
    recency = [r['features'].get('days_since_last_start') for r in rows]
    count = [r['features'].get('prior_start_count') for r in rows]
    distance = rows[0]['features'].get('target_distance_m')
    missing = any(r['features'].get(f) is None for r in rows for f in names)
    box = rows[fav[0]]['box']
    return {'favourite_probability': '<.4' if m.max() < .4 else '.4-.6' if m.max() < .6 else '>=.6',
            'entropy': '<.8' if entropy < .8 else '>=.8',
            'field_size': '<=6' if len(rows) <= 6 else str(len(rows)),
            'minimum_history': 'unknown' if None in count else '<5' if min(count) < 5 else '>=5',
            'maximum_recency': 'unknown' if None in recency else '<=21' if max(recency) <= 21 else '>21',
            'missing_base_input': str(missing),
            'disagreement_TV': '<.02' if np.abs(p-m).sum()/2 < .02 else '>=.02',
            'favourite_box': 'tied' if len(fav) > 1 else '1-2' if box <= 2 else '3-6' if box <= 6 else '7-8',
            'venue': rows[0]['venue'], 'distance': 'unknown' if distance is None else '<=400' if distance <= 400 else '>400',
            'grade': grade or 'unknown'}


def bootstrap_tables(races, draws=3000):
    days = sorted({r['race_date'] for r in races})
    rng = np.random.default_rng(20260929)
    weights = rng.multinomial(len(days), np.ones(len(days))/len(days), size=draws)
    specs = [('overall', 'all', model, races) for model in MODELS]
    for factor in races[0]['groups']:
        for value in sorted({r['groups'][factor] for r in races}):
            specs.append((factor, value, MODELS[0], [r for r in races if r['groups'][factor] == value]))
    table, family = [], []
    for factor, value, model, rr in specs:
        record = {'factor': factor, 'group': value, 'model': model, 'races': len(rr),
                  'dates': len({r['race_date'] for r in rr}),
                  'sparse': len(rr) < 20 or len({r['race_date'] for r in rr}) < 5,
                  'tied_favourites': sum(r['market']['top_ties'] > 1 for r in rr)}
        counts = np.array([sum(r['race_date'] == d for r in rr) for d in days])
        denominator = weights@counts
        for metric in ['ll', 'brier', 'accuracy']:
            record['market_'+metric] = float(np.mean([r['market'][metric] for r in rr]))
            record['model_'+metric] = float(np.mean([r[model][metric] for r in rr]))
        for metric in ['ll', 'brier']:
            diff = np.array([r['market'][metric]-r[model][metric] for r in rr])
            point = float(diff.mean())
            sums = np.array([sum(r['market'][metric]-r[model][metric] for r in rr if r['race_date'] == d) for d in days])
            bs = np.divide(weights@sums, denominator, out=np.full(draws, np.nan), where=denominator > 0)
            se = float(np.nanstd(bs, ddof=1))
            lodo = [float((diff.sum()-s)/(len(rr)-n)) for s,n in zip(sums,counts) if n and n < len(rr)]
            record[metric] = {'improvement': point, 'pointwise95': np.nanquantile(bs,[.025,.975]).tolist(),
                              'lodo_range': [min(lodo),max(lodo)] if lodo else None,
                              'bootstrap_nonempty': int(np.isfinite(bs).sum()),
                              'positive_races': int((diff > 1e-12).sum()), 'negative_races': int((diff < -1e-12).sum())}
            family.append((record[metric], se, np.abs(bs-point)/se if se > 1e-15 else np.zeros(draws)))
        table.append(record)
    # Common date draws preserve dependence across every reported proper-score contrast.
    maximum = np.nanmax(np.array([item[2] for item in family]), axis=0)
    critical = float(np.quantile(maximum, .95))
    for result, se, _ in family:
        result['simultaneous95'] = [result['improvement']-critical*se, result['improvement']+critical*se]
    return table, {'draws': draws, 'seed': 20260929, 'dates': days, 'contrasts': len(family), 'critical': critical,
                   'limitation': 'exploratory family adjustment, not correction for all earlier searches; sparse group bands unreliable'}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    ledger = out/'trial_ledger.jsonl'
    def event(name, **fields):
        with ledger.open('a') as handle:
            handle.write(json.dumps({'event': name, **fields}, sort_keys=True)+'\n')
    event('START', fits=0, protocol_sha256=sha((ROOT/'docs/research/market_explanation_20260929_plan.md').read_bytes()),
          code_sha256=sha(Path(__file__).read_bytes()))
    try:
        allowed, pins = load_scope()
        event('ACCESS_SCOPE_PASSED', development_races=331, runners=len(allowed), protected_decodes=0)
        expected = json.loads((ROOT/'docs/research/market_explanation_20260929_inputs.json').read_text())
        def read(path):
            payload = path.read_bytes()
            if sha(payload) != expected[str(path)]:
                raise ValueError('pinned research source changed: '+str(path))
            pins[str(path)] = sha(payload)
            return payload
        rows = gate_lines(read(SOURCE/'search_v1/outer_predictions.jsonl'), allowed)
        if len(rows) != 1251 or len({r['race_id'] for r in rows}) != 177:
            raise ValueError('outer prediction population changed')
        grouped = collections.defaultdict(list)
        for r in rows:
            grouped[r['race_id']].append(r)
        receipts = {fold: json.loads(read(SOURCE/f'search_v1/{fold}_models.json'))['base16']
                    for fold in {r['outer'] for r in rows}}
        provenance = json.loads(read(SOURCE/'foundation/form_provenance.json'))
        sources = {s['race_id']: s for s in provenance['sources']}
        races, contributions, forecast_examples = [], [], {}
        max_error = 0.0
        for rid, rr in sorted(grouped.items()):
            rr.sort(key=lambda r:r['box'])
            model = receipts[rr[0]['outer']]
            # The exact admitted pre-race sidecar supplies grade; no history files opened.
            source = sources[rid]
            path = Path(source['card_sidecar_path'])
            data = path.read_bytes()
            if sha(data) != source['card_sidecar_sha256']:
                raise ValueError('sidecar identity changed')
            pins[str(path)] = sha(data)
            metadata = json.loads(data)
            if metadata.get('metadata_is_leakage_safe') is not True:
                raise ValueError('pre-race metadata no longer qualified')
            grade = metadata.get('target_grade') or metadata.get('race_info', {}).get('grade')
            y = [r['y'] for r in rr]
            m = np.array([r['market'] for r in rr])
            inverse = 1/np.array([r['odds'] for r in rr])
            if not np.allclose(inverse/inverse.sum(), m, atol=1e-12, rtol=0):
                raise ValueError('market/odds mismatch')
            if len({r['capture'] for r in rr}) != 1 or any(not 2 <= r['capture_lead_minutes'] <= 10 for r in rr):
                raise ValueError('snapshot alignment')
            if any(r['features']['field_size'] != len(rr) for r in rr):
                raise ValueError('field size mismatch')
            race = {'race_id':rid, 'race_date':rr[0]['race_date'], 'outer':rr[0]['outer'],
                    'runner_count':len(rr), 'unoccupied_boxes': sorted(set(range(1,9))-{r['box'] for r in rr}),
                    'market':losses(y,m), 'groups':characteristics(rr,model['prep']['names'],grade)}
            for name in MODELS:
                race[name] = losses(y,[r['predictions'][name] for r in rr])
            d = decomposition(rr, model)
            for name, strength in [('refit_base16',1),('refit_half',.5)]:
                rebuilt = decomposition(rr,model,strength)['p']
                err = float(np.max(np.abs(rebuilt-np.array([r['predictions'][name] for r in rr]))))
                max_error = max(max_error,err)
                if err > 1e-12:
                    raise ValueError('retained coefficient reconstruction failed')
            runners = []
            for i,r in enumerate(rr):
                terms = dict(zip(d['names'],d['contributions'][i].tolist()))
                explanation = {'race_id':rid,'box':r['box'],'dog_token':r['dog_token'],'winner':bool(r['y']),
                    'market':r['market'],'model':r['predictions']['refit_base16'],
                    'linear_sum':float(d['z'][i]),'capped_residual':float(d['cap'][i]),
                    'cap_effect':float(d['cap_effect'][i]),'log_normalizer':d['log_normalizer'],
                    'log_probability_change':float(d['cap'][i]-d['log_normalizer']),
                    'contributions':terms,
                    'inputs':{f:r['features'].get(f) for f in model['prep']['names']}}
                contributions.append(explanation)
                runners.append(explanation)
            forecast_examples[rid] = runners
            races.append(race)
        event('RETAINED_RECONSTRUCTION_PASSED', races=len(races), runners=len(rows), max_absolute_error=max_error,
              unavailable_receipt='refit_box', fits=0)
        table, uncertainty = bootstrap_tables(races)
        dates = []
        for day in sorted({r['race_date'] for r in races}):
            rr = [r for r in races if r['race_date']==day]
            dates.append({'date':day,'races':len(rr), **{model:float(np.mean([r['market']['ll']-r[model]['ll'] for r in rr])) for model in MODELS}})
        effects = sorted(races,key=lambda r:r['market']['ll']-r['refit_base16']['ll'])
        gain = lambda r:r['market']['ll']-r['refit_base16']['ll']
        positive = [r for r in effects if gain(r)>0]
        negative = [r for r in effects if gain(r)<0]
        example_races = [effects[-1],effects[0],positive[len(positive)//2],negative[len(negative)//2]]
        examples = [{'selection':why,'race':r,'runners':forecast_examples[r['race_id']]}
                    for why,r in zip(['largest_help','largest_harm','median_positive','median_negative'],example_races)]
        influence = {'total_ll_improvement':sum(map(gain,races)),
            'positive_sum':sum(map(gain,positive)), 'negative_sum':sum(map(gain,negative)),
            'positive_races':len(positive),'negative_races':len(negative),
            'best_five':[{'race_id':r['race_id'],'gain':gain(r)} for r in effects[-5:][::-1]],
            'worst_five':[{'race_id':r['race_id'],'gain':gain(r)} for r in effects[:5]],
            'without_best_five':float(np.mean(list(map(gain,effects[:-5])))),
            'without_worst_five':float(np.mean(list(map(gain,effects[5:]))))}
        mechanical = []
        for name in receipts['period1']['prep']['names'] + ['missing::'+f for f in receipts['period1']['prep']['names']]:
            vals = [c['contributions'][name] for c in contributions]
            mechanical.append({'feature':name,'mean_absolute_linear_contribution':float(np.mean(np.abs(vals))),
                               'coefficients':{fold: model['beta'][(model['prep']['names']+['missing::'+f for f in model['prep']['names']]).index(name)] for fold,model in receipts.items()}})
        mechanical.sort(key=lambda item:-item['mean_absolute_linear_contribution'])
        for name,value in [('race_losses.json',races),('table.json',table),('uncertainty.json',uncertainty),
            ('date_losses.json',dates),('examples.json',examples),('influence.json',influence),
            ('mechanical_summary.json',mechanical),('input_hashes.json',pins),('base16_receipts.json',receipts),
            ('source_exclusions.json',json.loads(read(SOURCE/'foundation/exclusions.json')))]:
            write(out/name,value)
        with (out/'runner_contributions.jsonl').open('x') as handle:
            for r in contributions:
                handle.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
        write(out/'validation.json',{'races':len(races),'runners':len(rows),'dates':len(dates),
            'max_reconstruction_error':max_error,'new_fits':0,'protected_record_decodes':0,
            'missing_prediction_races':0,'training_only_development_races':331-len(races),
            'new_variants':0,'grade_semantics':'literal source category; no universal ordering',
            'scratch_status':'exact retained active roster; vacancy cause and scratch timing unknown'})
        event('COMPLETE', model_comparators=MODELS, subgroups=len(table)-3, fits=0, uncertainty_contrasts=uncertainty['contrasts'])
        write(out/'artifact_hashes.json',{p.name:sha(p.read_bytes()) for p in out.iterdir() if p.is_file()})
    except Exception as exc:
        event('FAILED', error=type(exc).__name__,detail=str(exc),fits=0)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
