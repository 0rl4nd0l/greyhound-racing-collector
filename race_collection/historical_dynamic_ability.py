"""Prior-date sequential ability on complete admitted historical race orders.

One interface, ``representations(events)``, emits three matched six-value form
representations. It never revises an earlier estimate with a later opponent
result or resolves an earlier identity using later cards. No provider access.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import math

ARMS = ('average', 'recency', 'dynamic')
FIELDS = ('ability', 'context_residual', 'latest_change', 'support',
          'context_support', 'uncertainty')
HALF_LIFE_DAYS = 30.
GLOBAL_PRIOR_STARTS = 3.
CONTEXT_PRIOR_STARTS = 5.


def _summary(history, day, context, arm):
    if not history:
        return dict(zip(FIELDS, (None, None, None, 0., 0. if context else None, None)))
    weights = [1. if arm == 'average' else 2. ** (-(day - h['day']).days / HALF_LIFE_DAYS)
               for h in history]
    total = sum(weights)
    mean = sum(w * h['performance'] for w, h in zip(weights, history)) / (GLOBAL_PRIOR_STARTS + total)
    selected = [(w, h) for w, h in zip(weights, history) if context and h['context'] == context]
    context_total = sum(w for w, _ in selected)
    residual = (sum(w * h['residual'] for w, h in selected) / (CONTEXT_PRIOR_STARTS + context_total)
                if context else None)
    variance = sum(w * (h['performance'] - mean) ** 2 for w, h in zip(weights, history)) / total
    # A weak unit-scale prior prevents zero estimated uncertainty after one run.
    uncertainty = math.sqrt((1. + variance) / (GLOBAL_PRIOR_STARTS + total))
    return dict(zip(FIELDS, (mean, residual, history[-1]['performance'] - mean,
                            total, context_total if context else None, uncertainty)))


def representations(events):
    """Return {arm: {(race_key, box): features}}, update audit and lineage.

    Events require date, key, context=(venue, distance) or None, and runners
    {box, native_id or None, finish}. Event labels are admitted complete orders.
    All predictions and opponent ratings on a date precede every update that
    date. Known track with unavailable distance remains missing context.
    """
    by_day = defaultdict(list)
    keys = set()
    for event in events:
        if event['key'] in keys:
            raise ValueError('DUPLICATE_RACE')
        keys.add(event['key'])
        positions = [r['finish'] for r in event['runners']]
        if not 2 <= len(positions) <= 8 or any(type(p) is not int for p in positions) or sorted(positions) != list(range(1, len(positions) + 1)):
            raise ValueError('COMPLETE_UNIQUE_ORDER_REQUIRED')
        context = event['context']
        if context is not None and (len(context) != 2 or not isinstance(context[0], str) or not context[0]
                                    or type(context[1]) not in (int, float) or not math.isfinite(context[1]) or context[1] <= 0):
            raise ValueError('INVALID_TRACK_DISTANCE_CONTEXT')
        boxes = [r['box'] for r in event['runners']]
        if len(set(boxes)) != len(boxes):
            raise ValueError('DUPLICATE_BOX')
        by_day[date.fromisoformat(event['date'])].append(event)
    output = {arm: {} for arm in ARMS}
    histories = {arm: defaultdict(list) for arm in ARMS}
    audit, lineage = Counter(), []
    for day, races in sorted(by_day.items()):
        # Ambiguous multiple starts cannot use guessed intraday sequencing.
        dog_counts = Counter(r['native_id'] for e in races for r in e['runners'] if r['native_id'])
        updates = {arm: [] for arm in ARMS}
        for event in sorted(races, key=lambda e: e['key']):
            runners, context = event['runners'], event['context']
            ids = [r['native_id'] for r in runners]
            eligible = all(ids) and all(dog_counts[native] == 1 for native in ids)
            audit['update_eligible_races' if eligible else 'update_excluded_identity_races'] += 1
            if not eligible:
                lineage.append({'race_key': event['key'], 'date': str(day), 'status': 'UPDATE_EXCLUDED_UNQUALIFIED_OR_AMBIGUOUS_IDENTITY'})
            for arm in ARMS:
                states = [_summary(histories[arm].get(native, []), day, context, arm)
                          if native else _summary([], day, context, arm) for native in ids]
                for i, (runner, state) in enumerate(zip(runners, states)):
                    output[arm][(event['key'], runner['box'])] = state
                    if not eligible:
                        continue
                    rank_performance = 1. - 2. * (runner['finish'] - 1.) / (len(runners) - 1.)
                    opponent = 0.
                    if arm == 'dynamic':
                        # Context effects are residuals around the dog's global
                        # rating. Both were recorded at the earlier event date.
                        opponent = sum((other['ability'] or 0.) + (other['context_residual'] or 0.)
                                       for j, other in enumerate(states) if j != i) / (len(runners) - 1)
                    performance = rank_performance + opponent
                    updates[arm].append((runner['native_id'], {
                        'day': day, 'context': context, 'performance': performance,
                        'residual': performance - (state['ability'] or 0.),
                        'source_race_key': event['key'],
                    }))
                    if arm == 'dynamic':
                        lineage.append({'race_key': event['key'], 'date': str(day), 'box': runner['box'],
                                        'native_id': runner['native_id'], 'status': 'PRIOR_DATE_UPDATE',
                                        'prior_support': state['support'], 'prior_ability': state['ability'],
                                        'prior_context_residual': state['context_residual'],
                                        'opponent_prior_mean': opponent, 'performance': performance,
                                        'qualified_opponents_with_prior_support': sum(j != i and s['support'] > 0 for j, s in enumerate(states))})
            audit['runner_appearances'] += len(runners)
        for arm in ARMS:
            for native, update in updates[arm]:
                histories[arm][native].append(update)
    audit['distinct_dates'] = len(by_day)
    audit['native_dogs_with_qualified_updates'] = len(histories['dynamic'])
    return output, dict(audit), lineage
