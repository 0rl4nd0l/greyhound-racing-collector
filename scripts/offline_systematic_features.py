"""As-of feature groups for the isolated systematic search; no target labels used."""
from __future__ import annotations
import collections
import json
import math
import statistics
from datetime import date
from pathlib import Path
from scripts import offline_form_packet as form
from scripts.offline_prediction_research import scalar, sha, write, FROZEN_FEATURES

GROUPS = {
    'draw': ['box_fraction', 'box_squared', 'inner_box', 'outer_box'],
    'draw_distance': ['box_distance', 'inner_short', 'outer_long'],
    'draw_track': ['__box_track__'],
    'recency': ['ewma_finish_h1', 'ewma_finish_h2', 'ewma_finish_h5', 'ewma_margin_h2', 'finish_recent_long_gap'],
    'margin_improvement': ['margin_recent_long_gap', 'margin_slope', 'finish_slope', 'improving_margin_weak_finish'],
    'times': ['time_field_gap', 'time_field_rank', 'time_best_field_gap', 'time_improvement', 'time_cv', 'same_layout_time_count'],
    'sectionals': ['section_field_gap', 'section_field_rank', 'section_best_field_gap', 'same_layout_section_count'],
    'pace_pressure': ['faster_neighbors', 'measured_neighbors', 'neighbor_section_gap', 'faster_field_fraction'],
    'grade_change': ['grade_changed', 'same_grade_recent_fraction', 'grade_unknown'],
    'experience': ['log_experience', 'log_layoff', 'layoff_21', 'layoff_60', 'limited_history'],
    'field_form': ['finish_field_rank', 'margin_field_rank', 'winrate_field_gap', 'field_history_coverage_x_logp'],
    'uncertainty': ['limited_x_logp', 'layoff_x_logp', 'time_missing_x_logp', 'section_missing_x_logp', 'form_dispersion'],
    'market_shape': ['log_market', 'market_squared', 'favorite_indicator', 'logp_x_concentration', 'logp_x_entropy', 'logp_x_overround'],
    'mechanism_interactions': ['inner_x_section_rank', 'favorite_x_pressure', 'outsider_x_margin_improvement'],
}
# These names are intentionally withheld from layout-sensitive comparisons.
AMBIGUOUS_LAYOUTS = {'QOT', 'RICH', 'MURR'}


def mean(values):
    values = [float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None


def slope(values):
    if len(values) < 3 or any(v is None for v in values):
        return None
    # Input newest first; reverse so negative slope denotes improvement.
    y = list(reversed(values)); xmean = (len(y) - 1) / 2
    return sum((i-xmean)*(v-statistics.mean(y)) for i,v in enumerate(y))/sum((i-xmean)**2 for i in range(len(y)))


def weighted(values, half_life):
    pairs = [(v, .5**(i/half_life)) for i,v in enumerate(values) if v is not None]
    return sum(v*w for v,w in pairs)/sum(w for _,w in pairs) if pairs else None


def positive(value):
    v = form.canonical.safe_float(value)
    return v if v is not None and v > 0 else None


def history_features(raw_rows, target_date, target_venue, distance, grade):
    """Reject ambiguous raw history membership; never repair from target results."""
    history, rejected = form.canonical.accepted_history(raw_rows, target_date)
    if rejected:
        raise ValueError('raw history membership differs from canonical accepted history')
    raw = sorted(raw_rows, key=lambda r: r['DATE'], reverse=True)
    if len(history) != len(raw) or len({r['DATE'] for r in raw}) != len(raw):
        raise ValueError('ambiguous same-day raw history order')
    if any(date.fromisoformat(r['DATE']) >= target_date for r in raw):
        raise ValueError('target/future history')
    finishes = [form.canonical.safe_float(r.get('PLC')) for r in raw]
    margins = [form.canonical.safe_float(r.get('MGN')) for r in raw]
    # Conservative venue-distance identity: merged layout aliases never qualify.
    same = [r for r in raw if target_venue not in AMBIGUOUS_LAYOUTS
            and form.canonical.canonical_venue(r.get('TRACK')) == target_venue
            and form._metres(r.get('DIST')) == distance]
    times = [v for r in same if (v := positive(r.get('TIME'))) is not None]
    sections = [v for r in same if (v := positive(r.get('1 SEC'))) is not None]
    rec, older = mean(margins[:2]), mean(margins[2:])
    gap = rec-older if rec is not None and older is not None else None
    fshort, flong = mean(finishes[:2]), mean(finishes[2:])
    features = {
        'ewma_finish_h1': weighted(finishes,1), 'ewma_finish_h2': weighted(finishes,2),
        'ewma_finish_h5': weighted(finishes,5), 'ewma_margin_h2': weighted(margins,2),
        'finish_recent_long_gap': fshort-flong if fshort is not None and flong is not None else None,
        'margin_recent_long_gap': gap, 'margin_slope': slope(margins[:5]),
        'finish_slope': slope(finishes[:5]),
        'improving_margin_weak_finish': -gap*max(0,(fshort or 0)-3) if gap is not None else None,
        '_time_mean': mean(times[:3]), '_time_best': min(times[:5]) if times else None,
        '_section_mean': mean(sections[:3]), '_section_best': min(sections[:5]) if sections else None,
        'same_layout_time_count': float(len(times)), 'same_layout_section_count': float(len(sections)),
        'time_improvement': (mean(times[:2])-mean(times[2:]))/mean(times) if len(times)>=3 else None,
        'time_cv': statistics.pstdev(times)/mean(times) if len(times)>=3 else None,
        'grade_changed': float(form.canonical.canonical_grade(raw[0].get('G')) != grade) if raw and grade!='__MISSING__' else None,
        'same_grade_recent_fraction': mean([float(form.canonical.canonical_grade(r.get('G')) == grade) for r in raw[:5]]) if grade!='__MISSING__' else None,
        'grade_unknown': float(grade=='__MISSING__'),
        'form_dispersion': statistics.pstdev(finishes[:5]) if len(finishes)>=3 and all(v is not None for v in finishes[:5]) else None,
    }
    return features


def relative(rows, source, gap_name, rank_name=None):
    available = [r['features'][source] for r in rows if r['features'].get(source) is not None]
    for r in rows:
        value = r['features'].get(source)
        r['features'][gap_name] = (value-statistics.median(available))/statistics.median(available) if value is not None and available and statistics.median(available) != 0 else None
        if rank_name:
            r['features'][rank_name] = sum(v<value for v in available)/max(1,len(available)-1) if value is not None else None


def build(prepared: Path, out: Path):
    # Read and pin restrictions before any labelled development record is decoded.
    protection = json.loads((prepared/'protected_records.json').read_text())
    audit = json.loads((prepared/'dataset_assessment.json').read_text())
    if min(k[:10] for k in protection['records']) <= '2026-07-09':
        raise ValueError('protected histories could enter development')
    data_path = prepared/'development.jsonl'
    if sha(data_path) != audit['development_sha256']:
        raise ValueError('prepared data hash changed')
    for line in data_path.open():
        if scalar(line,'race_date') > '2026-07-09':
            raise ValueError('outside authorised development before decoding labels')
    rows = [json.loads(line) for line in data_path.open()]
    provenance = json.loads((prepared/'form_provenance.json').read_text())
    sources = {s['race_id']:s for s in provenance['sources']}
    by_race = collections.defaultdict(list)
    for r in rows:
        by_race[r['race_id']].append(r)
    timing = []
    for rid, rr in by_race.items():
        src = sources[rid]
        card = form._verified(Path(src['card_source_path']),src['card_source_sha256'])
        metadata = json.loads(form._verified(Path(src['card_sidecar_path']),src['card_sidecar_sha256']))
        venue, _, grade, _ = form.canonical.target_metadata({'metadata':metadata},rid)
        distance = form._metres(metadata.get('target_distance') or metadata.get('race_info',{}).get('distance'))
        blocks = form.canonical.parse_form_blocks_bytes(card,source=rid)
        capture = form.canonical.capture_timestamp(metadata,require_timezone=True)
        jump = form.canonical.sidecar_jump_timestamp(metadata,rid)
        for r in rr:
            from datetime import datetime
            if jump != datetime.fromisoformat(r['jump']):
                raise ValueError('market/source jump mismatch')
            raw = blocks[r['dog_token']]
            if any(date.fromisoformat(v['DATE']) >= capture.date() for v in raw):
                raise ValueError('history not strictly before source capture date')
            r['layout_venue'] = venue
            f = r['features']
            f.update(history_features(raw,date.fromisoformat(r['race_date']),venue,distance,grade))
            logp = math.log(r['market']); box = r['box']; n = f['prior_start_count']; layoff = f['days_since_last_start']
            f.update({'box_fraction':(box-1)/7,'box_squared':((box-1)/7)**2,
                'inner_box':float(box<=2),'outer_box':float(box>=7),
                'box_distance':(box-4.5)*distance/500,
                'inner_short':float(box<=2 and distance<=400), 'outer_long':float(box>=7 and distance>=500),
                'log_experience':math.log1p(n), 'log_layoff':math.log1p(layoff) if layoff is not None else None,
                'layoff_21':float(layoff>=21) if layoff is not None else None,
                'layoff_60':float(layoff>=60) if layoff is not None else None,
                'limited_history':float(n<3), 'log_market':logp, 'market_squared':r['market']**2,
                'limited_x_logp':float(n<3)*logp,
                'layoff_x_logp':math.log1p(layoff)*logp if layoff is not None else None,
                'time_missing_x_logp':float(f['_time_mean'] is None)*logp,
                'section_missing_x_logp':float(f['_section_mean'] is None)*logp})
        for src,gap,rank in [('_time_mean','time_field_gap','time_field_rank'),('_time_best','time_best_field_gap',None),('_section_mean','section_field_gap','section_field_rank'),('_section_best','section_best_field_gap',None),('recent_finish_mean_3','finish_field_gap','finish_field_rank'),('recent_avg_margin_5','margin_field_gap','margin_field_rank')]:
            relative(rr,src,gap,rank)
        probs = [r['market'] for r in rr]
        concentration=sum(p*p for p in probs); entropy=-sum(p*math.log(p) for p in probs); overround=sum(1/r['odds'] for r in rr)
        for r in rr:
            f=r['features']; section=f['_section_mean']; others=[x for x in rr if x is not r and x['features']['_section_mean'] is not None]; adjacent=[x for x in others if abs(x['box']-r['box'])==1]
            f.update({'faster_neighbors':float(sum(x['features']['_section_mean']<section for x in adjacent)) if section is not None and adjacent else None,
                'measured_neighbors':float(len(adjacent)),
                'neighbor_section_gap':(section-mean([x['features']['_section_mean'] for x in adjacent]))/section if section is not None and adjacent else None,
                'faster_field_fraction':sum(x['features']['_section_mean']<section for x in others)/len(others) if section is not None and others else None,
                'winrate_field_gap':f['career_win_rate']-mean([x['features']['career_win_rate'] for x in rr]),
                'field_history_coverage_x_logp':mean([x['features']['limited_history'] for x in rr])*f['log_market'],
                'favorite_indicator':float(r['market']==max(probs)),
                'logp_x_concentration':f['log_market']*concentration,
                'logp_x_entropy':f['log_market']*entropy,
                'logp_x_overround':f['log_market']*overround,
                'inner_x_section_rank':f['inner_box']*f['section_field_rank'] if f['section_field_rank'] is not None else None,
                'favorite_x_pressure':float(r['market']==max(probs))*sum(x['features']['_section_mean']<section for x in adjacent) if section is not None and adjacent else None,
                'outsider_x_margin_improvement':float(r['odds']>=10)*f['improving_margin_weak_finish'] if f['improving_margin_weak_finish'] is not None else None})
            if any(v is not None and not math.isfinite(v) for v in f.values()):
                raise ValueError('nonfinite feature')
        timing.append({'race_id':rid,'card_capture':capture.isoformat(),'jump':jump.isoformat(),'history_max':max(v['DATE'] for r in rr for v in blocks[r['dog_token']]),'layout_venue':venue})
    out.mkdir(exist_ok=False)
    with (out/'features.jsonl').open('x') as handle:
        for r in rows:
            handle.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    write(out/'feature_groups.json',GROUPS)
    write(out/'feature_audit.json',{'races':len(by_race),'runners':len(rows),'timing':timing,
        'missingness':{f:sum(r['features'].get(f) is None for r in rows) for f in sorted({f for g in GROUPS.values() for f in g if not f.startswith('__')})},
        'layout_ambiguous_groups':sorted(AMBIGUOUS_LAYOUTS),'no_global_future_aggregates':True,
        'history_rule':'all raw histories before target AND before source capture date; no target SP/TIME/PIR used; no PIR features',
        'input_hash':sha(data_path),'feature_hash':sha(out/'features.jsonl')})
    return rows

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepared',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();rows=build(args.prepared,args.out);print(json.dumps({'races':len({r['race_id'] for r in rows}),'runners':len(rows)}))
