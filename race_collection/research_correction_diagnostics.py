"""Descriptive, exhaustive probability-correction accounting; never selects rules.

The input is an already authorized matched race population. This module neither
opens source evidence nor fits predictors. Conditions must be fixed before
looking at scores and attached without consulting the target outcome.
"""
from __future__ import annotations

from collections import defaultdict
import math


def _metric(runners, name):
    probabilities = [float(r['probabilities'][name]) for r in runners]
    if any(not math.isfinite(p) or not 0 < p <= 1 for p in probabilities):
        raise ValueError('finite positive probabilities required')
    if not math.isclose(sum(probabilities), 1., abs_tol=1e-8):
        raise ValueError('probabilities must sum to one within race')
    if sum(r['y'] for r in runners) != 1:
        raise ValueError('exactly one winner required')
    top = {i for i, p in enumerate(probabilities) if p == max(probabilities)}
    winner = next(i for i, r in enumerate(runners) if r['y'] == 1)
    return {'log_loss': -math.log(probabilities[winner]),
            'brier': sum((p - r['y']) ** 2 for p, r in zip(probabilities, runners)),
            'top_credit': float(winner in top) / len(top), 'top': top}


def summarize(races, *, reference, model):
    """Return every paired race and fixed strata, including ties and unchanged tops.

    Race fields: key, date, track, runners[{y, probabilities}], conditions mapping.
    Brier is the sum over runners averaged over races. Conditions are descriptive
    slices with their own coverage, not independent tests or predictive rules.
    """
    paired, seen = [], set()
    for race in races:
        if race['key'] in seen:
            raise ValueError('duplicate race')
        seen.add(race['key'])
        base, candidate = (_metric(race['runners'], name) for name in (reference, model))
        change = candidate['top_credit'] - base['top_credit']
        selection = ('unchanged_top' if candidate['top'] == base['top'] else
                     'changed_top_gained' if change > 0 else
                     'changed_top_lost' if change < 0 else 'changed_top_neither_gained_nor_lost')
        delta = candidate['log_loss'] - base['log_loss']
        paired.append({'key': race['key'], 'date': race['date'], 'track': race['track'],
                       'log_loss_delta': delta, 'brier_delta': candidate['brier'] - base['brier'],
                       'top_credit_delta': change, 'selection': selection,
                       'probability_effect': 'helps' if delta < 0 else 'hurts' if delta > 0 else 'equal',
                       'conditions': race.get('conditions', {})})
    if not paired:
        raise ValueError('empty matched population')

    def aggregate(rows):
        n = len(rows)
        return {'races': n, 'coverage': n / len(paired), 'dates': len({r['date'] for r in rows}),
                'log_loss_delta': sum(r['log_loss_delta'] for r in rows) / n,
                'brier_delta': sum(r['brier_delta'] for r in rows) / n,
                'top_credit_delta': sum(r['top_credit_delta'] for r in rows) / n,
                'helped': sum(r['probability_effect'] == 'helps' for r in rows),
                'hurt': sum(r['probability_effect'] == 'hurts' for r in rows),
                'equal': sum(r['probability_effect'] == 'equal' for r in rows)}

    groups = defaultdict(list)
    for row in paired:
        for kind in ('date', 'track', 'selection', 'probability_effect'):
            groups[(kind, str(row[kind]))].append(row)
        for condition, value in row['conditions'].items():
            groups[('condition:' + condition, str(value))].append(row)
    strata = [{'dimension': kind, 'value': value, **aggregate(rows)}
              for (kind, value), rows in sorted(groups.items())]
    dates = sorted({r['date'] for r in paired})
    leave_one_date_out = [{'omitted': day, **aggregate([r for r in paired if r['date'] != day])}
                         for day in dates if any(r['date'] != day for r in paired)]
    return {'reference': reference, 'model': model, 'status': 'EXPLORATORY_DESCRIPTIVE_ONLY',
            'overall': aggregate(paired), 'strata': strata,
            'leave_one_date_out': leave_one_date_out, 'paired_races': paired}
