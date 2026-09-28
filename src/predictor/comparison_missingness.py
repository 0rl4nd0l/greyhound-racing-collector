"""Prespecified outcome-free bounds for unresolved four-way result closure."""
from datetime import date, timedelta
import numpy as np


def paired_bounds(races, *, replicates=20000):
    """Bound unknown winners/voids; never impute missing results as random.

    An unresolved race may have any fractional winner distribution over its
    sealed field, or be void (zero difference). An outside-field result is not
    identifiable by these bounds and must block a broader claim separately.
    """
    output={}
    for candidate in ('residual_box','residual_half'):
        for reference in ('market','production'):
            bounds=[]
            for race in races:
                p=np.asarray(race['probabilities'][candidate],float)
                q=np.asarray(race['probabilities'][reference],float)
                if (len(p)!=len(q) or len(p)<2 or not np.isfinite(p).all() or not np.isfinite(q).all()
                        or (p<=0).any() or (q<=0).any() or not np.isclose(p.sum(),1) or not np.isclose(q.sum(),1)):
                    raise ValueError('invalid_common_probability')
                possible=np.column_stack((np.log(q/p),np.sum(p*p-q*q)-2*(p-q)))
                if race.get('outcome') is not None:
                    y=np.asarray(race['outcome'],float)
                    if len(y)!=len(p) or not np.isfinite(y).all() or (y<0).any() or not np.isclose(y.sum(),1):raise ValueError('invalid_fractional_outcome')
                    lo=hi=y@possible
                elif race.get('winner') is not None:
                    lo=hi=possible[race['winner']]
                elif race.get('official_void') is True:
                    lo=hi=np.zeros(2)
                else:
                    lo=np.minimum(possible.min(axis=0),0);hi=np.maximum(possible.max(axis=0),0)
                bounds.append([lo,hi])
            bounds=np.asarray(bounds)
            groups={}
            for unit in ('date','week'):
                keys=[r['date'] if unit=='date' else (date.fromisoformat(r['date'])-timedelta(days=date.fromisoformat(r['date']).weekday())).isoformat() for r in races]
                labels=sorted(set(keys));counts=np.array([keys.count(k) for k in labels])
                sums=np.array([bounds[[k==v for k in keys]].sum(axis=0) for v in labels])
                draws=np.random.default_rng(20260924).multinomial(len(labels),np.full(len(labels),1/len(labels)),size=replicates)
                means=np.einsum('bi,ijk->bjk',draws,sums)/(draws@counts)[:,None,None]
                interval=np.stack((np.quantile(means[:,0,:],.00625,axis=0),np.quantile(means[:,1,:],.99375,axis=0)),axis=1)
                groups[unit]={'blocks':len(labels),'simultaneous_outer_intervals':interval.tolist(),
                    'inferentially_usable':len(labels)>=(40 if unit=='date' else 12),
                    'both_upper_bounds_below_zero':bool(len(labels)>=(40 if unit=='date' else 12) and (interval[:,1]<0).all())}
            output[candidate+'-minus-'+reference]={'identified_mean_bounds':bounds.mean(axis=0).T.tolist(),'resampling':groups}
    return {'common_seals':len(races),'resolved':sum(r.get('winner') is not None or r.get('outcome') is not None for r in races),
        'unresolved':sum(r.get('winner') is None and r.get('outcome') is None and not r.get('official_void') for r in races),
        'paired':output,'interpretation':'Bounds assume a sealed-field outcome or void; technical exclusion and outside-field results are not missing at random.'}
